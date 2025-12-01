""""
NEW_VERSION="$(
python - <<'PY'
import re, sys, io, os
pyproject_path = os.environ["PYPROJECT_PATH"])"

with io.open(pyproject_path, "r", encoding="utf-8") as f:
    txt = f.read()

# Find [project] version = "..."
# We'll replace the FIRST occurrence of version = "..."
m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"\s*$', txt)
if not m:
    print("", end="")
    sys.exit(0)

cur = m.group(1)
devm = re.search(r'^(.*)\.dev(\d+)$', cur)
if devm:
    base, n = devm.group(1), int(devm.group(2))
    new = f"{base}.dev{n+1}"
else:
    # no .dev suffix -> append .dev0 (per your request)
    new = cur + ".dev0"

new_txt = txt[:m.start(1)] + new + txt[m.end(1):]
with io.open(pyproject_path, "w", encoding="utf-8") as f:
    f.write(new_txt)

print(new, end="")
PY
PYPROJECT_PATH="${PYPROJECT}"
)"

if [[ -z "${NEW_VERSION}" ]]; then
  echo "✗ Could not find a static [project].version in ${PYPROJECT}" >&2
  exit 1
fi"""


def main(path):
    import toml

    print(f"Loading pyproject from: {path}")
    data = toml.load(path)

    version = data.get("project", {}).get("version")
    if version is None or not version:
        version = "0.0.0.dev0"
    elif ".dev" in version:
        base, _, n = version.rpartition(".dev")
        if n.isdigit():
            n = int(n) + 1
        else:
            n = 0
        version = f"{base}.dev{n}"
    else:
        version = f"{version}.dev0"

    data["project"]["version"] = version
    with open(path, "w") as f:
        toml.dump(data, f)
        print(f"Bumped version to: {version}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bump version script")
    parser.add_argument("-path", type=str, help="path to pyproject.toml")
    pyproject_path = parser.parse_args().path
    main(pyproject_path)
