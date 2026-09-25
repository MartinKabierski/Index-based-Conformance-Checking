#!/usr/bin/env python3
import subprocess
import re
import os
from pathlib import Path


def _find_project_root_from_class(runner_class_file: Path) -> Path:
    """
    Läuft vom Runner.class nach oben und sucht target/classes.
    """
    runner_class_file = runner_class_file.resolve()

    for p in [runner_class_file] + list(runner_class_file.parents):
        if (p / "target" / "classes").exists():
            return p

    raise RuntimeError(
        f"target/classes nicht gefunden über: {runner_class_file}"
    )


def _build_classpath(project_root: Path) -> str:
    """
    Classpath = target/classes + lib/*.jar (falls vorhanden)
    """
    cp = [str(project_root / "target" / "classes")]

    lib_dir = project_root / "lib"
    if lib_dir.exists():
        cp.extend(str(j) for j in lib_dir.glob("*.jar"))

    sep = ";" if os.name == "nt" else ":"
    return sep.join(cp)


def _detect_main_class_from_class_path(runner_class_file: Path, project_root: Path) -> str:
    """
    Rekonstruiert den Klassennamen aus dem Pfad.
    """
    classes_root = project_root / "target" / "classes"
    rel = runner_class_file.resolve().relative_to(classes_root)
    return ".".join(rel.with_suffix("").parts)


def run_trie_conformance(
        runner_class: str,
        proxy_log: str,
        sample_log: str,
        java_bin: str = "/usr/bin/java"
):
    """
    Startet Runner.class mit zwei Log-Dateien,
    parst Runtime (ms) und Fitness aus dem Output.
    """
    runner_file = Path(runner_class).resolve()

    # Projektroot finden
    project_root = _find_project_root_from_class(runner_file)

    # Classpath bauen
    classpath = _build_classpath(project_root)

    # Klassennamen bestimmen
    main_class = _detect_main_class_from_class_path(runner_file, project_root)

    # Logfiles prüfen
    proxy_log_path = Path(proxy_log).resolve()
    sample_log_path = Path(sample_log).resolve()

    if not proxy_log_path.exists():
        raise FileNotFoundError(f"Proxy-Log nicht gefunden: {proxy_log_path}")
    if not sample_log_path.exists():
        raise FileNotFoundError(f"Sample-Log nicht gefunden: {sample_log_path}")

    # Java-Kommando
    cmd = [
        java_bin,
        "-cp",
        classpath,
        main_class,
        str(proxy_log_path),
        str(sample_log_path)
    ]

    # Subprozess ausführen
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    output = result.stdout

    # Runtime extrahieren
    m_time = re.search(
        r"Time taken.*?(\d+)\s+milliseconds",
        output, flags=re.IGNORECASE
    )
    # Fitness extrahieren
    m_fit = re.search(
        r"Overall fitness\s*=\s*([0-9.eE+-]+)",
        output, flags=re.IGNORECASE
    )

    if not m_time:
        raise RuntimeError("Runtime nicht im Output gefunden:\n" + output)
    if not m_fit:
        raise RuntimeError("Fitness nicht im Output gefunden:\n" + output)

    return float(m_fit.group(1)), int(m_time.group(1))
