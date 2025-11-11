#!/usr/bin/env python3
"""
Synchronise Jupyter notebooks from an external repository into the blog.

Usage
-----
python scripts/export_notebooks.py \
  --input external-notebooks/math_for_ml \
  --output src/data/blog/notebooks \
  --public public/notebooks
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Iterable

DEFAULT_AUTHOR = "Kalpesh Chavan"
DEFAULT_DESCRIPTION = "Lecture notes converted from Jupyter notebooks."
DEFAULT_TAGS = ["notebook", "math", "ml"]
IGNORED_FOLDERS = {"MNIST", "MNIST_data", ".ipynb_checkpoints", "media"}


def run_nbconvert(notebook: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
  """Convert a notebook to markdown using nbconvert and return the md path."""
  output_dir.mkdir(parents=True, exist_ok=True)
  slug = notebook.stem
  subprocess.run(
      [
          sys.executable,
          "-m",
          "jupyter",
          "nbconvert",
          "--to",
          "markdown",
          "--output",
          slug,
          "--output-dir",
          str(output_dir),
          str(notebook),
      ],
      check=True,
  )
  return output_dir / f"{slug}.md"


def slugify(name: str) -> str:
  return (
      name.lower()
      .replace(" ", "-")
      .replace("_", "-")
      .replace("/", "-")
  )


def ensure_frontmatter(markdown_path: pathlib.Path, title: str) -> None:
  """Inject frontmatter if the markdown file doesn't already have any."""
  content = markdown_path.read_text(encoding="utf-8")
  if content.startswith("---"):
    return

  now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  lines = [
      "---",
      f"title: {title}",
      f"author: {DEFAULT_AUTHOR}",
      f"description: {DEFAULT_DESCRIPTION}",
      f"pubDatetime: {now}",
      "modDatetime:",
      "draft: true",
      "tags:",
  ]
  lines.extend(f"  - {tag}" for tag in DEFAULT_TAGS)
  lines.append("---\n")
  markdown_path.write_text("\n".join(lines) + content, encoding="utf-8")


def move_assets(slug: str, markdown_path: pathlib.Path, output_assets_root: pathlib.Path) -> None:
  """Move accompanying *_files directory into the public asset folder."""
  source_dir = markdown_path.parent / f"{markdown_path.stem}_files"
  text = markdown_path.read_text(encoding="utf-8")

  if source_dir.exists():
    for file in source_dir.iterdir():
      if file.is_file():
        target = output_assets_root / file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    shutil.rmtree(source_dir)
    text = text.replace(f"{markdown_path.stem}_files/", "/notebooks/media/")

  text = text.replace("](media/", "](/notebooks/media/")
  text = text.replace("](./media/", "](/notebooks/media/")
  text = text.replace("](../media/", "](/notebooks/media/")
  text = text.replace('src="./media/', 'src="/notebooks/media/')
  text = text.replace("src='../media/", "src='/notebooks/media/")
  text = text.replace("src='./media/", "src='/notebooks/media/")
  text = text.replace("src=\"media/", "src=\"/notebooks/media/")
  markdown_path.write_text(text, encoding="utf-8")


def copy_media_folder(source_root: pathlib.Path, public_root: pathlib.Path) -> None:
  media_dir = source_root / "media"
  if not media_dir.exists():
    return
  target = public_root / "media"
  if target.exists():
    shutil.rmtree(target)
  shutil.copytree(media_dir, target)


def iter_notebooks(root: pathlib.Path) -> Iterable[pathlib.Path]:
  for path in root.rglob("*.ipynb"):
    if any(part in IGNORED_FOLDERS for part in path.parts):
      continue
    yield path


def main() -> None:
  parser = argparse.ArgumentParser(description="Export notebooks into markdown blog posts.")
  parser.add_argument("--input", default="external-notebooks/math_for_ml", type=pathlib.Path)
  parser.add_argument("--output", default="src/data/blog/notebooks", type=pathlib.Path)
  parser.add_argument("--public", default="public/notebooks", type=pathlib.Path)
  args = parser.parse_args()

  args.output.mkdir(parents=True, exist_ok=True)
  args.public.mkdir(parents=True, exist_ok=True)

  copy_media_folder(args.input, args.public)

  for notebook in iter_notebooks(args.input):
    rel_parent = notebook.parent.relative_to(args.input)
    target_dir = args.output / rel_parent
    markdown_path = run_nbconvert(notebook, target_dir)
    slug = slugify(notebook.stem)
    ensure_frontmatter(markdown_path, title=notebook.stem.replace("_", " ").title())
    move_assets(slug, markdown_path, args.public)


if __name__ == "__main__":
  main()
