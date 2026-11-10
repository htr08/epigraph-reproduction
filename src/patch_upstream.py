"""Compatibility patches for the upstream EpiGraph code (https://github.com/sj584/EpiGraph).

The upstream code was written against an older torch_geometric. With a recent version
(e.g. on Kaggle) three small things break. This script fixes them in place and is
idempotent (safe to run many times).

    python src/patch_upstream.py --upstream /kaggle/working/EpiGraph

1. model.py    : `data.node_attrs` collides with a method of the new PyG `Data` class
                 -> use `data['node_attrs']`.
2. evaluate.py : old checkpoints store GATConv weights as `lin_src` / `lin_dst`,
                 new PyG uses a single `lin` -> rename keys when loading.
3. train.py    : new PyG `Data` objects are not hashable, so `set(X) - set(X_test)` fails
                 -> build the training list by index (same samples, same original order).
"""
import argparse
import os
import re


def patch_model(path):
    s = open(path).read()
    s = s.replace("data.node_attrs", "data['node_attrs']")
    open(path, "w").write(s)


def patch_evaluate(path):
    s = open(path).read()
    if "lin_src" in s:
        return
    pat = re.compile(
        r"^([ \t]*)model\.load_state_dict\(torch\.load\(f'\{direc\}/\{pt\}'\)\)", re.M
    )
    assert pat.search(s), "could not find the load_state_dict line in evaluate.py"

    def rep(m):
        i = m.group(1)
        return (
            i + "sd = torch.load(f'{direc}/{pt}', map_location='cpu')\n"
            + i + "sd = {k.replace('lin_src', 'lin'): v for k, v in sd.items() if 'lin_dst' not in k}\n"
            + i + "model.load_state_dict(sd)"
        )

    open(path, "w").write(pat.sub(rep, s))


def patch_train(path):
    s = open(path).read()
    old = "X_train = list(set(X) - set(X_test))"
    new = "X_train = [d for i, d in enumerate(X) if i < test_sta or i > test_end]"
    if old in s:
        open(path, "w").write(s.replace(old, new))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="EpiGraph", help="path to the cloned upstream repo")
    a = ap.parse_args()
    patch_model(os.path.join(a.upstream, "model.py"))
    patch_evaluate(os.path.join(a.upstream, "evaluate.py"))
    patch_train(os.path.join(a.upstream, "train.py"))
    print("upstream patched OK:", a.upstream)


if __name__ == "__main__":
    main()
