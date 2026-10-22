#!/usr/bin/env python3
"""
biblioteca.py — Extractor dirigido por instrucción en lenguaje natural

Usa Firecrawl (self-host, local) para convertir una URL o un blog entero
a markdown limpio, y Ollama (modelo local, p.ej. qwen2.5:14b-instruct)
para interpretar una instrucción en lenguaje natural sobre qué extraer,
cómo nombrar cada archivo y en qué carpeta guardarlo.

Requisitos previos:
  - Firecrawl self-host corriendo en http://localhost:3002
  - Ollama corriendo en http://localhost:11434 con el modelo ya descargado:
        ollama pull qwen2.5:14b-instruct
  - pip install firecrawl requests

Uso:
  Página suelta:
    python biblioteca.py --url "https://ejemplo.com/pagina" \
        --instruccion "descarga las recetas de remedios naturales que encuentres" \
        --carpeta "biblioteca/remedios-naturales"

  Blog entero (recorre varias páginas):
    python biblioteca.py --url "https://ejemplo.com" --crawl --limite 20 \
        --instruccion "descarga los artículos sobre movilidad eléctrica" \
        --carpeta "biblioteca/movilidad-urbana"
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from urllib.parse import urljoin, urlparse
from datetime import datetime

import requests
from firecrawl import Firecrawl

FIRECRAWL_URL = "http://localhost:3002"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def slugify(texto: str) -> str:
    """Convierte un título en un nombre de archivo seguro."""
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    texto = texto.lower().strip()
    texto = re.sub(r"[^a-z0-9]+", "-", texto)
    texto = re.sub(r"-+", "-", texto).strip("-")
    return texto[:80] if texto else "sin-titulo"


def asegurar_carpeta(ruta: str):
    os.makedirs(ruta, exist_ok=True)


# ---------------------------------------------------------------------------
# Paso 1: obtener contenido con Firecrawl
# ---------------------------------------------------------------------------

def obtener_paginas(url: str, crawl: bool, limite: int) -> list[dict]:
    """
    Devuelve una lista de páginas: [{"url": ..., "titulo": ..., "markdown": ...}, ...]
    """
    app = Firecrawl(api_key="no-necesaria", api_url=FIRECRAWL_URL)

    if crawl:
        print(f"[Firecrawl] Recorriendo {url} (límite {limite} páginas)...")
        job = app.crawl(url, limit=limite, scrape_options={"formats": ["markdown"]})
        paginas = []
        for pagina in job.data:
            paginas.append({
                "url": getattr(pagina.metadata, "source_url", None) or url,
                "titulo": getattr(pagina.metadata, "title", None) or "",
                "markdown": pagina.markdown or "",
            })
        print(f"[Firecrawl] {len(paginas)} páginas obtenidas.")
        return paginas
    else:
        print(f"[Firecrawl] Descargando {url}...")
        doc = app.scrape(url, formats=["markdown"])
        return [{
            "url": url,
            "titulo": doc.metadata.title if doc.metadata else "",
            "markdown": doc.markdown or "",
        }]


# ---------------------------------------------------------------------------
# Modo --buscar: localizar los enlaces relevantes a partir de la página de entrada
# ---------------------------------------------------------------------------

PROMPT_ENLACES = """Eres un asistente que elige enlaces de una página web según la instrucción del usuario.
Recibes una lista numerada de enlaces (URL y texto del enlace) de una página. Clasifica SOLO los relevantes:

- "articulos": enlaces que llevan directamente a un artículo/entrada/receta individual que encaja con la instrucción.
- "secciones": enlaces a una categoría, listado o índice (p.ej. /blog/, /categoria/recetas/, /archivo/) donde probablemente haya más \
artículos que encajen con la instrucción.

Ignora menús genéricos, aviso legal, cookies, contacto, redes sociales, etiquetas, paginación irrelevante y artículos de otros temas.

Devuelve SIEMPRE un JSON válido, sin texto adicional: {"articulos": [números], "secciones": [números]}"""


def _enlaces_candidatos(url_base: str, markdown: str, links: list[str]) -> list[tuple[str, str]]:
    """Enlaces del mismo dominio, sin duplicados, con su texto de anclaje si lo hay."""
    host = urlparse(url_base).netloc
    textos = {}
    for texto, href in re.findall(r"\[((?:[^\[\]]|\\.)*)\]\(([^)\s]+)\)", markdown):
        limpio = re.sub(r"[\\#*_`🛴🚲⚡🛡️→]+|\s+", " ", texto).strip()
        if limpio and len(limpio) > len(textos.get(href, "")):
            textos[href] = limpio[:150]
    vistos, resultado = set(), []
    for href in links:
        href = urljoin(url_base, href).split("#")[0]
        if urlparse(href).netloc != host or href.rstrip("/") == url_base.rstrip("/") or href in vistos:
            continue
        if re.search(r"\.(jpg|jpeg|png|gif|webp|svg|pdf|zip|css|js|xml)$", href, re.I):
            continue
        vistos.add(href)
        resultado.append((href, textos.get(href, "")))
    return resultado[:200]


_RE_PAGINA = re.compile(r"(/(page|paged|pagina|pag|p)/\d+/?$)|([?&](page|paged|pagina|pag|p)=\d+)", re.I)
_RE_SIGUIENTE = re.compile(r"^\W*(siguiente|next|older|más antiguas?|entradas anteriores|página siguiente)\W*$|^[»›→>]+$", re.I)


def _base_seccion(url: str) -> str:
    """Ruta de la sección sin el sufijo de paginación (/blog/page/2/ -> /blog)."""
    ruta = re.sub(r"/(page|paged|pagina|pag|p)/\d+/?$", "", urlparse(url).path, flags=re.I)
    return ruta.rstrip("/")


def _separar_paginacion(pagina: str, candidatos: list[tuple[str, str]]):
    """Separa los enlaces de paginación DE LA MISMA sección (/blog/ -> /blog/page/2/, ?page=2,
    'Siguiente', '»') del resto, para seguirlos sin gastar la selección del modelo."""
    base = _base_seccion(pagina)
    paginacion, resto = [], []
    for href, texto in candidatos:
        pagina_num = _RE_PAGINA.search(href) or _RE_PAGINA.search(texto or "")
        siguiente = _RE_SIGUIENTE.match(texto or "")
        misma = _base_seccion(href) == base
        if (pagina_num or siguiente) and misma:
            paginacion.append(href)
        else:
            resto.append((href, texto))
    return paginacion, resto


def _elegir_enlaces(candidatos: list[tuple[str, str]], instruccion: str) -> tuple[list[str], list[str]]:
    lista = "\n".join(f"{i}. {href} — {texto}" for i, (href, texto) in enumerate(candidatos))
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT_ENLACES},
            {"role": "user", "content": f'Instrucción del usuario: "{instruccion}"\n\nEnlaces:\n{lista}'},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    if resp.status_code == 404:
        sys.exit(f"[error] Ollama no tiene el modelo '{OLLAMA_MODEL}'. Mira 'ollama list' y usa --modelo <nombre>.")
    resp.raise_for_status()
    try:
        datos = json.loads(resp.json()["message"]["content"])
    except json.JSONDecodeError:
        return [], []

    def urls(clave):
        indices = [i for i in datos.get(clave, []) if isinstance(i, int) and 0 <= i < len(candidatos)]
        return [candidatos[i][0] for i in indices]

    return urls("articulos"), urls("secciones")


def buscar_paginas(url: str, instruccion: str, limite: int, profundidad: int = 2, max_paginas: int = 10) -> list[dict]:
    """Parte de 'url', deja que el modelo elija los enlaces relevantes (y secciones donde
    seguir buscando) y devuelve el markdown de hasta 'limite' páginas de artículos.
    En cada sección sigue la paginación (hasta 'max_paginas' páginas por sección)."""
    app = Firecrawl(api_key="no-necesaria", api_url=FIRECRAWL_URL)
    visitadas = {url.rstrip("/")}
    pendientes = [url]
    articulos: list[str] = []
    cache: dict[str, dict] = {}

    for nivel in range(profundidad):
        siguientes = []
        for pagina in pendientes:
            cola = [pagina]
            leidas = 0
            while cola and leidas < max_paginas and len(articulos) < limite:
                actual = cola.pop(0)
                leidas += 1
                print(f"[Buscar] Leyendo enlaces de {actual}...")
                doc = app.scrape(actual, formats=["markdown", "links"])
                if actual == pagina:
                    cache[pagina] = {
                        "url": pagina,
                        "titulo": getattr(doc.metadata, "title", None) or "",
                        "markdown": doc.markdown or "",
                    }
                candidatos = _enlaces_candidatos(actual, doc.markdown or "", doc.links or [])
                paginacion, candidatos = _separar_paginacion(actual, candidatos)
                for p in paginacion:
                    if p.rstrip("/") not in visitadas:
                        visitadas.add(p.rstrip("/"))
                        cola.append(p)
                if not candidatos:
                    continue
                arts, secs = _elegir_enlaces(candidatos, instruccion)
                print(f"  {len(arts)} artículo(s) y {len(secs)} sección(es) candidatos.")
                for a in arts:
                    if a.rstrip("/") not in visitadas:
                        visitadas.add(a.rstrip("/"))
                        articulos.append(a)
                for sec in secs:
                    if sec.rstrip("/") not in visitadas:
                        visitadas.add(sec.rstrip("/"))
                        siguientes.append(sec)
        pendientes = siguientes
        if not pendientes or len(articulos) >= limite:
            break

    # Las secciones visitadas también pueden ser un artículo completo (p.ej. una página de tema):
    # se analizan igualmente; el tipo de página decide si se guardan.
    secciones = [p for p in cache if p.rstrip("/") != url.rstrip("/")]
    articulos = (articulos + [p for p in secciones if p not in articulos])[:limite]
    print(f"[Buscar] Analizando {len(articulos)} página(s) candidatas...")
    paginas = []
    for a in articulos:
        if a in cache:
            paginas.append(cache[a])
            continue
        try:
            doc = app.scrape(a, formats=["markdown"])
        except Exception as e:
            print(f"  [aviso] No se pudo descargar {a}: {e}")
            continue
        paginas.append({
            "url": a,
            "titulo": getattr(doc.metadata, "title", None) or "",
            "markdown": doc.markdown or "",
        })
    return paginas


# ---------------------------------------------------------------------------
# Paso 2: interpretar la página con Ollama según la instrucción del usuario
# ---------------------------------------------------------------------------

PROMPT_SISTEMA = """Eres un asistente que IDENTIFICA elementos específicos dentro del contenido de una \
página web en markdown, siguiendo la instrucción exacta del usuario. NO copias ni resumes el contenido: \
solo localizas dónde empieza cada elemento. El programa se encarga de extraer el texto.

Devuelve SIEMPRE un JSON válido con esta forma exacta, sin texto adicional antes ni después:

{
  "tipo_pagina": "articulo | varios | listado | otro",
  "elementos": [
    {
      "nombre": "Nombre corto del elemento (p.ej. el nombre de la receta o el título del artículo)",
      "inicio": "Copia EXACTA, carácter por carácter, de la línea de la página donde empieza ese elemento (normalmente su título)"
    }
  ]
}

Reglas importantes:
- "tipo_pagina" describe LA PÁGINA ENTERA, no un trozo: "articulo" = la página ES un único artículo/receta completo (su título principal es ese elemento); \
"varios" = la página contiene varios elementos independientes del tipo que pide el usuario, uno tras otro (p.ej. varias recetas distintas); "listado" = portada, categoría, buscador o índice que muestra títulos con resumen y enlaces \
("Leer más", "Leer la guía") a otras páginas; "otro" = aviso legal, contacto, etc. Si es "listado" u "otro", devuelve "elementos": [].
- Los apartados, subtítulos o secciones internas de UN mismo artículo NO son elementos distintos: si la página es un solo artículo con sus apartados, tipo_pagina es "articulo" y devuelves un único elemento (el artículo entero).
- Juzga por el CONTENIDO de la página, no solo por los títulos: un título que suene a lo que pide el usuario no basta si debajo solo hay un resumen de dos líneas o un enlace.
- Solo incluye elementos cuyo contenido completo esté en esta página y coincidan claramente con lo que pide el usuario.
- NO incluyas anuncios, resúmenes, extractos o listados que solo enlazan al contenido completo (p.ej. "Leer más →"). Si la página es un listado de entradas, devuelve "elementos": [].
- NO incluyas insignias, avisos cortos, etiquetas de verificación de precio ni fragmentos sin contenido propio (p.ej. "Verificación pendiente", "Precio sujeto a cambios", "En oferta").
- Si hay varios elementos de la misma marca o categoría pero son productos/artículos distintos (p.ej. varios modelos de una misma marca), usa en "nombre" el nombre COMPLETO y específico que los distinga entre sí (marca + modelo + lo que los diferencie), nunca solo la marca genérica repetida.
- Si ninguno coincide, devuelve "elementos": [].
- "inicio" debe ser una línea que exista literalmente en la página.
- Devuelve los elementos en el mismo orden en que aparecen en la página.
- No añadas comentarios ni explicaciones fuera del JSON.
"""


def _normalizar(linea: str) -> str:
    """Normaliza una línea para compararla: quita cabeceras markdown, énfasis,
    numeración/viñetas iniciales, emojis y símbolos sueltos."""
    texto = linea.strip()
    texto = texto.lstrip("#").strip()
    texto = re.sub(r"[*_`]", "", texto)                  # negrita/cursiva/código
    texto = re.sub(r"^[-•▪►]\s*", "", texto)              # viñetas
    texto = re.sub(r"^\d+[\.\)]\s*", "", texto)           # "1. " / "2) "
    texto = re.sub(r"[^\w\sáéíóúñüÁÉÍÓÚÑÜ]", "", texto, flags=re.UNICODE)  # emojis/símbolos
    texto = re.sub(r"\s+", " ", texto).strip().lower()
    return texto


def _localizar(objetivo: str, normalizadas: list[str], desde: int) -> int | None:
    """Busca 'objetivo' entre las líneas normalizadas a partir de 'desde'.
    Primero coincidencia exacta; si no hay, coincidencia por subcadena en
    cualquiera de los dos sentidos (tolera que el modelo abrevie o añada algo)."""
    if not objetivo:
        return None
    for i in range(desde, len(normalizadas)):
        if normalizadas[i] == objetivo:
            return i
    for i in range(desde, len(normalizadas)):
        if normalizadas[i] and (objetivo in normalizadas[i] or normalizadas[i] in objetivo):
            return i
    return None


def _palabras(texto: str) -> set[str]:
    return {p for p in _normalizar(texto).split() if len(p) > 2}


def _titulo_real(markdown: str, titulo: str) -> str:
    """Título principal de la página: primer H1 (ATX o setext) y, si no hay, el de la metadata."""
    lineas = markdown.splitlines()
    for i, linea in enumerate(lineas):
        if linea.startswith("# "):
            return linea[2:].strip()
        if i + 1 < len(lineas) and linea.strip() and re.fullmatch(r"=+", lineas[i + 1].strip()):
            return linea.strip()
    return re.split(r"\s[|–—]\s", titulo or "")[0].strip()


def _parece_listado(markdown: str) -> bool:
    """Muchos encabezados que son enlaces + 'Leer más' = portada/índice, no un artículo."""
    encabezados_enlace = len(re.findall(r"^#{2,4}\s*\[", markdown, re.M))
    leer_mas = len(re.findall(r"^\s*(leer|ver)\b.*(→|»)\s*$", markdown, re.M | re.I))
    return encabezados_enlace >= 3 or leer_mas >= 3


def trocear(markdown: str, elementos: list[dict], tipo_pagina: str = "", titulo: str = "", min_lineas: int = 0) -> list[dict]:
    """Rellena 'contenido' con el texto ORIGINAL de la página (sin pasar por el modelo)."""
    if not elementos or tipo_pagina in ("listado", "otro"):
        return []

    # Un único elemento: solo si la página ES ese artículo se guarda entera,
    # con el título real de la página (no el que invente el modelo).
    if len(elementos) == 1:
        nombre = elementos[0].get("nombre", "")
        if tipo_pagina == "articulo" and not _parece_listado(markdown):
            return [{"nombre": _titulo_real(markdown, titulo) or nombre, "contenido": markdown}]
        print(f"  [omitido] '{nombre}' no es el contenido principal de esta página (es un resumen/enlace).")
        return []

    lineas = markdown.splitlines()
    normalizadas = [_normalizar(l) for l in lineas]
    posiciones = []
    cursor = 0
    for el in elementos:
        objetivo = _normalizar(el.get("inicio", ""))
        idx = _localizar(objetivo, normalizadas, cursor)
        if idx is None:
            print(f"  [aviso] No se localizó el inicio de '{el.get('nombre', '')}', se omite.")
            continue
        posiciones.append((idx, el.get("nombre", "")))
        cursor = idx + 1

    resultado = []
    for n, (idx, nombre) in enumerate(posiciones):
        fin = posiciones[n + 1][0] if n + 1 < len(posiciones) else len(lineas)
        resultado.append({"nombre": nombre, "contenido": "\n".join(lineas[idx:fin])})

    # Si la mayoría de los "elementos" son trozos demasiado cortos, en realidad son los
    # apartados de un único artículo: se guarda la página entera en vez de sus fragmentos.
    if min_lineas and len(resultado) > 1:
        cortos = sum(1 for r in resultado if sum(1 for l in r["contenido"].splitlines() if l.strip()) < min_lineas)
        if cortos * 2 >= len(resultado) and not _parece_listado(markdown):
            nombre = _titulo_real(markdown, titulo)
            if nombre:
                print("  [aviso] Los elementos eran apartados de un único artículo; se guarda la página entera.")
                return [{"nombre": nombre, "contenido": markdown}]
    return resultado


def extraer_elementos(markdown: str, instruccion: str, url_origen: str, titulo: str = "", min_lineas: int = 0) -> list[dict]:
    """Llama a Ollama para interpretar la página según la instrucción del usuario."""

    # Ollama corre en local y sin límite de tokens de pago, pero conviene no
    # mandar páginas absurdamente largas de una sola vez.
    contenido = markdown[:40000]

    mensaje_usuario = f"""Instrucción del usuario: "{instruccion}"

URL de origen: {url_origen}

Contenido de la página (markdown):
---
{contenido}
---

Identifica los elementos que coincidan con la instrucción, siguiendo el formato JSON indicado."""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": mensaje_usuario},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1},
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    if resp.status_code == 404:
        sys.exit(f"[error] Ollama no tiene el modelo '{OLLAMA_MODEL}'. Mira 'ollama list' y usa --modelo <nombre> o 'ollama pull {OLLAMA_MODEL}'.")
    resp.raise_for_status()
    texto = resp.json()["message"]["content"]

    try:
        datos = json.loads(texto)
    except json.JSONDecodeError:
        print(f"  [aviso] Ollama no devolvió JSON válido para {url_origen}, se omite.")
        return []

    return trocear(markdown, datos.get("elementos", []), str(datos.get("tipo_pagina", "")).lower(), titulo, min_lineas)


# ---------------------------------------------------------------------------
# Paso 3: guardar cada elemento
# ---------------------------------------------------------------------------

def _nucleo(texto_archivo: str) -> str:
    """Quita la cabecera (fuente/fecha) para comparar solo el contenido real."""
    return texto_archivo.split("---\n\n", 1)[-1].strip()


def _mismo_contenido(ruta: str, contenido_nuevo: str) -> bool:
    try:
        with open(ruta, encoding="utf-8") as f:
            existente = _nucleo(f.read())
    except OSError:
        return False
    # Comparación por los primeros 300 caracteres: suficiente para detectar
    # si es realmente el mismo artículo/receta/modelo, sin exigir coincidencia exacta.
    return existente[:300].strip() == contenido_nuevo.strip()[:300].strip()


def guardar_elemento(elemento: dict, carpeta: str, url_origen: str, min_lineas: int = 0) -> str | None:
    nombre = elemento.get("nombre", "").strip() or "sin-titulo"
    contenido = elemento.get("contenido", "").strip()

    if not contenido:
        return None

    lineas = sum(1 for l in contenido.splitlines() if l.strip())
    if lineas < min_lineas:
        print(f"  [omitido] Demasiado corto ({lineas} líneas < {min_lineas}): {nombre}")
        return None

    base = slugify(nombre)
    nombre_archivo = base + ".md"
    ruta = os.path.join(carpeta, nombre_archivo)

    sufijo = 2
    while os.path.exists(ruta):
        if _mismo_contenido(ruta, contenido):
            print(f"  [omitido] Ya existe (mismo contenido): {ruta}")
            return None
        # Mismo nombre pero contenido distinto -> no lo pisamos, lo numeramos.
        nombre_archivo = f"{base}-{sufijo}.md"
        ruta = os.path.join(carpeta, nombre_archivo)
        sufijo += 1

    with open(ruta, "w", encoding="utf-8") as f:
        f.write(f"# {nombre}\n\n")
        f.write(f"Fuente: {url_origen}\n")
        f.write(f"Capturado: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")
        f.write(contenido)
        f.write("\n")

    print(f"  [guardado] {ruta}")
    return ruta


def actualizar_indice(carpeta: str, instruccion: str, nuevos: list[str]):
    """Escribe index_<instruccion>.md con los archivos hallados con esa instrucción:
    los de esta ejecución más los que ya figuraban en el índice de la misma instrucción."""
    ruta_indice = os.path.join(carpeta, f"index_{slugify(instruccion)}.md")

    archivos = []
    if os.path.exists(ruta_indice):
        with open(ruta_indice, encoding="utf-8") as f:
            archivos = re.findall(r"\]\(([^)]+\.md)\)", f.read())
    for ruta in nuevos:
        nombre = os.path.basename(ruta)
        if nombre not in archivos:
            archivos.append(nombre)
    archivos = sorted(a for a in archivos if os.path.exists(os.path.join(carpeta, a)))

    with open(ruta_indice, "w", encoding="utf-8") as f:
        f.write(f"# Índice — {os.path.basename(carpeta)}\n\n")
        f.write(f"Instrucción: {instruccion}\n")
        f.write(f"Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for archivo in archivos:
            with open(os.path.join(carpeta, archivo), encoding="utf-8") as g:
                primera = g.readline().lstrip("# ").strip()
            f.write(f"- [{primera or archivo[:-3]}]({archivo})\n")
    print(f"[índice] Actualizado {ruta_indice} ({len(archivos)} elementos)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global OLLAMA_MODEL
    parser = argparse.ArgumentParser(description="Biblioteca personal dirigida por instrucción")
    parser.add_argument("--url", required=True, help="URL de la página o del blog")
    parser.add_argument("--instruccion", required=True, help="Qué quieres extraer, en lenguaje natural")
    parser.add_argument("--carpeta", required=True, help="Carpeta donde guardar los resultados")
    parser.add_argument("--crawl", action="store_true", help="Recorrer varias páginas del sitio")
    parser.add_argument("--buscar", action="store_true",
                        help="Partiendo de --url, el modelo localiza los enlaces a artículos que encajan con la instrucción y los descarga")
    parser.add_argument("--limite", type=int, default=10, help="Máximo de páginas a recorrer (crawl) o de artículos a descargar (buscar)")
    parser.add_argument("--min-lineas", type=int, default=15,
                        help="Mínimo de líneas no vacías para guardar un elemento (descarta anuncios/resúmenes). 0 = sin filtro")
    parser.add_argument("--modelo", default=OLLAMA_MODEL, help="Modelo de Ollama (ver 'ollama list')")
    args = parser.parse_args()
    OLLAMA_MODEL = args.modelo

    asegurar_carpeta(args.carpeta)

    if args.buscar:
        paginas = buscar_paginas(args.url, args.instruccion, args.limite)
    else:
        paginas = obtener_paginas(args.url, args.crawl, args.limite)

    guardados = []
    for i, pagina in enumerate(paginas, 1):
        print(f"\n[{i}/{len(paginas)}] Analizando {pagina['url']}...")
        if not pagina["markdown"].strip():
            print("  [omitido] Página sin contenido markdown.")
            continue

        elementos = extraer_elementos(pagina["markdown"], args.instruccion, pagina["url"], pagina["titulo"], args.min_lineas)
        print(f"  Ollama encontró {len(elementos)} elemento(s) que coinciden.")

        for elemento in elementos:
            ruta = guardar_elemento(elemento, args.carpeta, pagina["url"], args.min_lineas)
            if ruta:
                guardados.append(ruta)

    if guardados:
        actualizar_indice(args.carpeta, args.instruccion, guardados)

    print(f"\nHecho. {len(guardados)} elemento(s) nuevo(s) guardado(s) en '{args.carpeta}'.")


if __name__ == "__main__":
    main()
