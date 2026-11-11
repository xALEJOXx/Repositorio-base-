"""Generar versiones y hashes para el entorno Linux del contenedor."""

import hashlib
import subprocess
import sys
import tempfile
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile


def main():
    raiz = Path(__file__).resolve().parent

    with tempfile.TemporaryDirectory() as temporal:
        carpeta = Path(temporal)

        # Descargar solo un wheel compatible por dependencia.
        subprocess.run(
            [
                sys.executable, "-m", "pip", "download",
                "--only-binary=:all:",
                "--dest", str(carpeta),
                "-r", str(raiz / "requirements-runtime.txt"),
            ],
            check=True,
        )

        requisitos = []

        for archivo in sorted(carpeta.glob("*.whl")):
            with ZipFile(archivo) as wheel:
                metadatos = [
                    nombre for nombre in wheel.namelist()
                    if nombre.endswith(".dist-info/METADATA")
                ]
                if len(metadatos) != 1:
                    raise ValueError(f"Metadatos inesperados: {archivo.name}")

                datos = BytesParser().parsebytes(
                    wheel.read(metadatos[0])
                )

            with archivo.open("rb") as contenido:
                huella = hashlib.file_digest(
                    contenido, "sha256"
                ).hexdigest()

            requisitos.append(
                f"{datos['Name']}=={datos['Version']} "
                f"--hash=sha256:{huella}"
            )

        if not requisitos:
            raise ValueError("No se descargaron dependencias.")

        destino = raiz / "requirements-runtime.lock"
        provisional = raiz / "requirements-runtime.lock.tmp"

        provisional.write_text(
            "# Generado para Linux y Python 3.13; incluye dependencias indirectas.\n"
            + "\n".join(sorted(requisitos, key=str.lower))
            + "\n",
            encoding="utf-8",
        )
        provisional.replace(destino)

        print(
            f"Lock creado correctamente: {len(requisitos)} paquetes.",
            flush=True,
        )


if __name__ == "__main__":
    main()