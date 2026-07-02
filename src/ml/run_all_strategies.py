import subprocess
import sys

def run_strategy(strategy: str):
    print(f"\n=============================================")
    print(f"      INICIANDO ESTRATEGIA: {strategy.upper()}      ")
    print(f"=============================================\n")
    
    cmd = [sys.executable, "-u", "src/orchestrator.py", "--strategy", strategy, "--skip-tests"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    if process.returncode != 0:
        print(f"\n[ERROR] Estrategia {strategy} fallo con codigo {process.returncode}")
        sys.exit(process.returncode)
    else:
        print(f"\n[EXITO] Estrategia {strategy} completada satisfactoriamente.")

if __name__ == "__main__":
    run_strategy("half")
    run_strategy("total")
    print("\n[GOAL COMPLETADO] Pipeline completo ejecutado.")
