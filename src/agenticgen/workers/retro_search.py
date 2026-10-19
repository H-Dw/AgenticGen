"""Run pinned upstream MCTS with exact disk stock; leave all chemical scoring unchanged.

Only the two stock-loading sites are replaced in memory. The upstream files are
not edited. Fail if their expected AST no longer matches; never silently fall back
to a toy/smaller inventory. CLI, templates, filter weights and algorithm stay upstream.
"""

import ast
import inspect
import os
import sys
import textwrap

from stock_index import DiskMolInventory, SQLiteStock


def replace_stock_loading(source, *, constructor):
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    body, changes = [], 0
    for node in function.body:
        if (
            constructor
            and isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "instock_list" for t in node.targets)
        ):
            node.value = ast.parse(
                "SQLiteStock(os.environ['AGENTICGEN_STOCK_INDEX'])", mode="eval"
            ).body
            changes += 1
        if (
            not constructor
            and isinstance(node, ast.With)
            and any(
                isinstance(n, ast.Attribute) and n.attr == "inventory_smiles_file"
                for n in ast.walk(node)
            )
        ):
            body.extend(
                ast.parse(
                    "if config.canonicalize_inventory:\n"
                    "    raise ValueError('Indexed stock requires canonicalize_inventory=false')\n"
                    "mol_inventory = DiskMolInventory(os.environ['AGENTICGEN_STOCK_INDEX'])"
                ).body
            )
            changes += 1
            continue
        if (
            not constructor
            and isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "mol_inventory" for t in node.targets)
        ):
            changes += 1
            continue
        body.append(node)
    if changes != (1 if constructor else 2):
        raise ValueError("Pinned upstream stock-loading AST changed; review adapter")
    function.body = body
    return ast.fix_missing_locations(tree)


def main():
    import syntheseus.cli.search as search
    import syntheseus.reaction_prediction.inference.simpretro as model

    if not os.environ.get("AGENTICGEN_STOCK_INDEX"):
        raise ValueError("Indexed search requires the configured stock database")
    for original, constructor, namespace in (
        (model.SimpRetro.__init__, True, vars(model)),
        (search.run_from_config, False, vars(search)),
    ):
        namespace.update(SQLiteStock=SQLiteStock, DiskMolInventory=DiskMolInventory, os=os)
        tree = replace_stock_loading(inspect.getsource(original), constructor=constructor)
        exec(compile(tree, inspect.getsourcefile(original), "exec"), namespace)
        if constructor:
            model.SimpRetro.__init__ = namespace["__init__"]
    return search.main(sys.argv[2:])


if __name__ == "__main__":
    main()
