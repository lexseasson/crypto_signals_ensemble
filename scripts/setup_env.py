#!/usr/bin/env python3
"""Utility to bootstrap a local .env file from the committed template."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_env(force: bool = False) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template = repo_root / ".env.example"
    target = repo_root / ".env"

    if not template.exists():
        raise FileNotFoundError(
            "El archivo .env.example no existe. Asegúrate de haber clonado el repositorio correctamente."
        )

    if target.exists() and not force:
        print("Ya existe un archivo .env. Usa --force para sobrescribirlo si lo necesitas.")
        return

    shutil.copy(template, target)
    print("Archivo .env creado a partir de .env.example. Completa tus credenciales locales antes de ejecutar la app.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea o actualiza el archivo .env local a partir del template controlado"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribe el .env existente en caso de que ya exista en el directorio del proyecto.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    copy_env(force=args.force)
    