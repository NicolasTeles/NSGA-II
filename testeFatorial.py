#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orquestrador dos experimentos fatoriais do NSGA-II (dieta PHDI x aceitabilidade).

O que ele faz de diferente do orquestrador antigo
  * Cada execução (conjunto x seed) grava numa pasta própria (saida/exec/cXXX_eYY/).
    A pasta só "existe de verdade" quando a execução terminou (é criada como .tmp e renomeada).
    => se o processo morrer (sessão tmux encerrada, queda de energia...), nada fica corrompido,
       e é só rodar o MESMO comando de novo: ele pula o que já terminou e continua do ponto parado.
  * As execuções são ordenadas por SEED (todas as configurações da seed 1, depois da seed 2...).
    => uma interrupção deixa seeds inteiras completas: os dados continuam balanceados e pareáveis.
  * Nada é apagado automaticamente (o antigo clear() foi removido).
  * Ao final (ou ao ser interrompido com Ctrl+C) junta tudo em arquivos únicos
    (pareto_final.csv, evolucao_agregada.csv, pareto_snapshots.csv, execucoes.csv)
    e escreve completude.csv mostrando o que foi planejado x o que existe.

Execução sequencial (padrão, 1 processo) — não interfere em quem usa o servidor:
    python3 testeFatorial.py --run 3 --saida resultados_run3

Para testar rápido (12 execuções pequenas):
    python3 testeFatorial.py --run teste --saida resultados_teste

Opcional: paralelizar (use com cuidado em servidor compartilhado):
    python3 testeFatorial.py --run 3 --saida resultados_run3 --workers 4

Rodar "solto" do terminal (sobrevive ao fechar a sessão, se o servidor permitir):
    nohup python3 testeFatorial.py --run 3 --saida resultados_run3 > run3.log 2>&1 &
"""
import os

# Evita que o numpy abra várias threads por processo (educado com servidor compartilhado).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import datetime
import itertools
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback

# ----------------------------------------------------------------------------------
# Definição dos experimentos (mesmas grades das runs anteriores)
#   eixos : (nome_do_parametro, niveis) na ORDEM em que o número do conjunto é gerado
#   fixos : parâmetros constantes na run
# Parâmetros: N, G, C (chance de crossover), M (chance de mutação), CI, DI (índices SBX/mutação)
# ----------------------------------------------------------------------------------
RUNS = {
    "1": dict(nome="Run 1",
              eixos=[("N", [100, 300]), ("G", [200, 500]), ("C", [0.7, 1.0]), ("M", [0.1, 0.3])],
              fixos={"CI": 20, "DI": 20}, seeds=list(range(30))),
    "2": dict(nome="Run 2",
              eixos=[("CI", [2, 5, 10, 15, 20, 40, 55]), ("DI", [5, 10, 15, 20, 35, 45, 65])],
              fixos={"N": 300, "G": 500, "C": 1.0, "M": 0.1}, seeds=list(range(30))),
    "3": dict(nome="Run 3",
              eixos=[("N", [200, 300, 400]), ("G", [400, 500, 600]), ("CI", [5, 10, 15]), ("DI", [15, 20, 35])],
              fixos={"C": 1.0, "M": 0.1}, seeds=list(range(30, 60))),
    "teste": dict(nome="Teste rápido",
                  eixos=[("N", [40, 60]), ("G", [30]), ("CI", [5, 15]), ("DI", [20])],
                  fixos={"C": 1.0, "M": 0.1}, seeds=[1, 2, 3]),
}
PARAMS = ["N", "G", "C", "M", "CI", "DI"]


def agora():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def gerar_jobs(run):
    """Lista de execuções, ordenada por seed (laço externo) e depois por conjunto."""
    cfg = RUNS[run]
    nomes = [e[0] for e in cfg["eixos"]]
    combos = list(itertools.product(*[e[1] for e in cfg["eixos"]]))
    jobs = []
    for execucao, seed in enumerate(cfg["seeds"], start=1):
        for conjunto, combo in enumerate(combos, start=1):
            p = dict(cfg["fixos"])
            p.update(dict(zip(nomes, combo)))
            jobs.append(dict(conjunto=conjunto, execucao=execucao, seed=seed, **{k: p[k] for k in PARAMS}))
    return jobs, len(combos)


def nome_job(j):
    return f"c{j['conjunto']:03d}_e{j['execucao']:02d}"


def versao_codigo():
    """Hash do git (com '+alteracoes' se há mudanças não commitadas), p/ rastrear qual código gerou os dados."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        if h.returncode != 0:
            return "sem-git"
        sujo = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10).stdout.strip()
        return h.stdout.strip() + ("+alteracoes" if sujo else "")
    except Exception:
        return "sem-git"


# ----------------------------------------------------------------------------------
# Execução de uma tarefa (roda no processo principal ou em um worker)
# ----------------------------------------------------------------------------------
def executar_job(args):
    job, saida, snapshot_a_cada, evolucao_completa, versao = args
    destino = os.path.join(saida, "exec", nome_job(job))
    tmp = destino + ".tmp"
    if os.path.isdir(destino):
        return ("pulado", job, None)

    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    t0, inicio = time.time(), agora()
    try:
        import NSGA2  # import tardio: cada processo carrega o problema uma vez
        info = NSGA2.main(
            NUM_INDIVIDUOS=job["N"], GERACOES=job["G"],
            CHANCE_CROSSOVER=job["C"], CHANCE_MUTACAO=job["M"],
            SEED=job["seed"], CROSSOVER_INDEX=job["CI"], DISTRIBUTION_INDEX=job["DI"],
            execucao=job["execucao"], conjunto=job["conjunto"],
            pasta_saida=tmp, salvar_graficos=False, salvar_log_txt=False, verbose=False,
            snapshot_a_cada=snapshot_a_cada, evolucao_completa=evolucao_completa, retornar_info=True,
        )
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        return ("erro", job, traceback.format_exc())

    meta = dict(job)
    meta.update(tempo_cpu_s=round(info["tempo"], 3), tempo_wall_s=round(time.time() - t0, 3),
                n_pareto=info["n_pareto"], tam_pop_final=info["tam_pop_final"],
                inicio=inicio, fim=agora(), versao_codigo=versao,
                host=platform.node(), python=platform.python_version(), pid=os.getpid())
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(meta, f)
    os.replace(tmp, destino)   # renomeação atômica: a execução passa a "existir" só agora
    return ("ok", job, meta)


# ----------------------------------------------------------------------------------
# Consolidação e completude
# ----------------------------------------------------------------------------------
def _juntar(nome_arquivo, pastas, destino):
    """Concatena (em streaming) os CSVs de cada execução, conferindo que o cabeçalho é igual."""
    cab, n_arq = None, 0
    with open(destino, "w", newline="") as out:
        for p in pastas:
            caminho = os.path.join(p, nome_arquivo)
            if not os.path.exists(caminho):
                continue
            with open(caminho, newline="") as f:
                header = f.readline()
                if cab is None:
                    cab = header
                    out.write(header)
                elif header != cab:
                    raise RuntimeError(f"Cabeçalho diferente em {caminho}: arquivos de versões distintas misturados?")
                shutil.copyfileobj(f, out)
                n_arq += 1
    if n_arq == 0 and os.path.exists(destino):
        os.remove(destino)
    return n_arq


def consolidar(saida, jobs, n_conjuntos):
    base = os.path.join(saida, "exec")
    pastas = sorted(os.path.join(base, d) for d in os.listdir(base)
                    if os.path.isdir(os.path.join(base, d)) and not d.endswith(".tmp")) if os.path.isdir(base) else []
    print(f"\n[consolidação] {len(pastas)} execuções concluídas encontradas.")
    if not pastas:
        return
    for nome in ("pareto_final.csv", "evolucao_agregada.csv", "pareto_snapshots.csv", "pareto_evolucao.csv"):
        n = _juntar(nome, pastas, os.path.join(saida, nome))
        if n:
            mb = os.path.getsize(os.path.join(saida, nome)) / 1e6
            print(f"  {nome:24s} {n:5d} execuções  {mb:9.1f} MB")

    metas = []
    for p in pastas:
        with open(os.path.join(p, "meta.json")) as f:
            metas.append(json.load(f))
    if metas:
        with open(os.path.join(saida, "execucoes.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metas[0].keys()))
            w.writeheader()
            w.writerows(metas)

    # completude: planejado x existente, por conjunto
    feitos = {(m["conjunto"], m["execucao"]) for m in metas}
    por_conj = {}
    for j in jobs:
        d = por_conj.setdefault(j["conjunto"], dict(planejadas=0, concluidas=0, faltam=[]))
        d["planejadas"] += 1
        if (j["conjunto"], j["execucao"]) in feitos:
            d["concluidas"] += 1
        else:
            d["faltam"].append(j["execucao"])
    with open(os.path.join(saida, "completude.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Conjunto", "planejadas", "concluidas", "execucoes_faltando"])
        for c, d in sorted(por_conj.items()):
            w.writerow([c, d["planejadas"], d["concluidas"], " ".join(map(str, d["faltam"]))])
    completos = sum(1 for d in por_conj.values() if d["concluidas"] == d["planejadas"])
    print(f"  completude: {len(feitos)}/{len(jobs)} execuções; conjuntos completos: {completos}/{n_conjuntos}")
    if len(feitos) < len(jobs):
        print("  (há execuções faltando: rode o mesmo comando de novo para continuar de onde parou)")


# ----------------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, choices=list(RUNS), help="qual experimento: 1, 2, 3 ou teste")
    ap.add_argument("--saida", required=True, help="pasta de resultados (será criada; pode ser reaproveitada para retomar)")
    ap.add_argument("--workers", type=int, default=1,
                    help="processos em paralelo (padrão 1 = sequencial). Em servidor compartilhado, combine com o responsável.")
    ap.add_argument("--nice", type=int, default=10, help="prioridade baixa (0-19) para não atrapalhar outros usuários (padrão 10)")
    ap.add_argument("--snapshot-a-cada", type=int, default=100, help="salva o front a cada N gerações (0 = não salva)")
    ap.add_argument("--evolucao-completa", action="store_true", help="grava também o pareto_evolucao.csv ANTIGO (enorme; só depuração)")
    ap.add_argument("--max-execucoes", type=int, default=None, help="roda no máximo K execuções e para (para testar/retomar aos poucos)")
    ap.add_argument("--sem-consolidar", action="store_true")
    ap.add_argument("--consolidar-apenas", action="store_true", help="não roda nada; só junta o que já existe")
    ap.add_argument("--limpar", action="store_true", help="APAGA a pasta de saída antes de começar (pede confirmação)")
    a = ap.parse_args()

    cfg = RUNS[a.run]
    jobs, n_conj = gerar_jobs(a.run)
    saida = os.path.abspath(a.saida)

    if a.limpar and os.path.isdir(saida):
        if input(f"Apagar TUDO em {saida}? digite 'sim': ").strip().lower() != "sim":
            sys.exit("Cancelado.")
        shutil.rmtree(saida)
    os.makedirs(os.path.join(saida, "exec"), exist_ok=True)

    # a pasta se ignora sozinha no git (o '*' também ignora este próprio arquivo)
    gi = os.path.join(saida, ".gitignore")
    if not os.path.exists(gi):
        with open(gi, "w") as f:
            f.write("*\n")

    # manifesto: protege contra retomar uma pasta com outro planejamento
    manifesto = os.path.join(saida, "manifest.csv")
    linhas = [",".join(["conjunto", "execucao", "seed"] + PARAMS)] + [
        ",".join(str(j[k]) for k in ["conjunto", "execucao", "seed"] + PARAMS) for j in jobs]
    if os.path.exists(manifesto):
        if open(manifesto).read().splitlines() != linhas:
            sys.exit("ERRO: esta pasta de saída tem um planejamento (manifest.csv) diferente do atual.\n"
                     "Use outra pasta em --saida (ou --limpar, se quiser recomeçar).")
    else:
        with open(manifesto, "w") as f:
            f.write("\n".join(linhas) + "\n")

    if a.consolidar_apenas:
        consolidar(saida, jobs, n_conj)
        return

    try:
        os.nice(a.nice)
    except Exception:
        pass

    versao = versao_codigo()
    pendentes = [j for j in jobs if not os.path.isdir(os.path.join(saida, "exec", nome_job(j)))]
    if a.max_execucoes is not None:
        pendentes = pendentes[:a.max_execucoes]
    ja = len(jobs) - len([j for j in jobs if not os.path.isdir(os.path.join(saida, "exec", nome_job(j)))])

    print(f"[{agora()}] {cfg['nome']}: {n_conj} conjuntos x {len(cfg['seeds'])} seeds = {len(jobs)} execuções")
    print(f"  já concluídas: {ja} | a executar agora: {len(pendentes)} | processos: {a.workers} | código: {versao}")
    if a.workers > 1:
        print(f"  [atenção] paralelo: esta máquina tem {os.cpu_count()} CPUs; confira a carga (comando 'uptime') "
              f"e use no máximo metade dos núcleos livres.")

    log = open(os.path.join(saida, "log.txt"), "a", buffering=1)   # buffering=1: grava a cada linha
    log.write(f"\n=== {agora()} início: {len(pendentes)} execuções, workers={a.workers}, código={versao} ===\n")
    tarefas = [(j, saida, a.snapshot_a_cada, a.evolucao_completa, versao) for j in pendentes]

    feitos, erros, t_ini = 0, 0, time.time()

    def registrar(res):
        nonlocal feitos, erros
        status, job, extra = res
        if status == "pulado":
            return
        if status == "erro":
            erros += 1
            msg = f"[{agora()}] ERRO {nome_job(job)}:\n{extra}"
            print(msg)
            log.write(msg + "\n")
            return
        feitos += 1
        decorrido = time.time() - t_ini
        restante = (len(tarefas) - feitos - erros) * decorrido / max(feitos, 1) / max(a.workers, 1)
        linha = (f"[{agora()}] {nome_job(job)} ok | N={job['N']} G={job['G']} CI={job['CI']} DI={job['DI']} "
                 f"seed={job['seed']} | cpu={extra['tempo_cpu_s']:.0f}s | {feitos}/{len(tarefas)} | "
                 f"ETA ~{restante/3600:.1f} h")
        print(linha)
        log.write(linha + "\n")

    try:
        if a.workers <= 1:
            for t in tarefas:
                registrar(executar_job(t))
        else:
            import multiprocessing as mp
            with mp.Pool(processes=a.workers) as pool:
                for res in pool.imap_unordered(executar_job, tarefas):
                    registrar(res)
    except KeyboardInterrupt:
        print("\nInterrompido (Ctrl+C). Nada se perdeu: rode o mesmo comando para continuar.")
        log.write(f"[{agora()}] interrompido pelo usuário\n")

    log.write(f"=== {agora()} fim: {feitos} concluídas, {erros} com erro ===\n")
    log.close()
    print(f"[{agora()}] terminou: {feitos} concluídas, {erros} com erro.")
    if erros:
        print("  Execuções com erro não foram gravadas; rode o mesmo comando para tentar de novo.")

    if not a.sem_consolidar:
        consolidar(saida, jobs, n_conj)


if __name__ == "__main__":
    main()