# Guía Rápida: Cómo aceptar cambios y commits

Esta guía explica tres maneras de incorporar cambios propuestos por un mentor o colaborar con tus estudiantes/instructores usando Git. Está pensada para quienes trabajan con este repositorio educativo y quieren mantener un flujo limpio sin exponer credenciales.

## 1. Aceptar cambios desde GitHub (interfaz web)
1. Abre la Pull Request en GitHub.
2. Revisa la pestaña **Files changed** para inspeccionar el diff.
3. Si estás conforme, pulsa el botón verde **Merge pull request**.
4. Confirma con **Confirm merge**. GitHub creará automáticamente un commit de fusión en la rama principal.
5. Luego sincroniza tu copia local con `git pull origin main` para traer el merge.

> 💡 Usa la opción **Squash and merge** si quieres mantener un historial más compacto; esto combina todos los commits de la PR en uno solo.

## 2. Traer los cambios con la línea de comandos
1. Asegúrate de estar en la rama adecuada (`main` o la rama de trabajo):
   ```bash
   git checkout main
   ```
2. Descarga los commits más recientes del repositorio remoto:
   ```bash
   git pull origin main
   ```
   Esto actualiza tu copia local con el merge que aprobaste en GitHub.
3. Si estás revisando una rama de trabajo antes de fusionarla, puedes inspeccionarla localmente:
   ```bash
   git fetch origin nombre-de-la-rama
   git checkout nombre-de-la-rama
   ```
4. Revisa los cambios (`git status`, `git diff`) y decide si deseas hacer ajustes adicionales antes de fusionar.

## 3. Aplicar un parche específico con `git apply`
Si recibes un archivo `.patch` o deseas tomar un fragmento concreto de código:
1. Guarda el parche en tu máquina, por ejemplo `cambio.patch`.
2. Desde la raíz del repositorio ejecuta:
   ```bash
   git apply cambio.patch
   ```
3. Verifica los archivos modificados con `git status` y `git diff`.
4. Añade los archivos que quieras incluir en el commit:
   ```bash
   git add ruta/al/archivo.py
   ```
5. Crea el commit con un mensaje descriptivo:
   ```bash
   git commit -m "Explica brevemente el cambio"
   ```
6. Sube el commit a GitHub:
   ```bash
   git push origin nombre-de-tu-rama
   ```

## Buenas prácticas adicionales
* Revisa siempre `git status` para confirmar que `.env` u otros archivos sensibles no estén en *staging*.
* Usa ramas separadas (`git checkout -b feature/riesgo`) para cada mejora y facilita la revisión.
* Antes de fusionar, ejecuta las pruebas recomendadas (por ejemplo `python -m compileall .` o `pytest`) y adjunta el resultado en la PR.
* Si necesitas descartar un parche aplicado, utiliza `git reset --hard` **solo** si estás seguro de que no perderás trabajo importante.

Siguiendo estos pasos podrás aceptar, aplicar y versionar contribuciones sin perder el control del repositorio ni comprometer credenciales.