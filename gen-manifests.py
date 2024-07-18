

def lookup_pipeline_py(source_folder: str):
    """ lookup_pipeline_py lookup *pipeline.py and strip its parent dir
    For example: aaa/bbb/pipeline.py with source_folder=aaa
    would have outcome bbb/pipeline.py
    """
    import glob

    lookup_pattern = "{}/**/*pipeline.py".format(source_folder)
    print("lookup pattern:", lookup_pattern)
    return glob.glob(lookup_pattern, recursive=True)

def dynamic_import_and_compile(pipeline_path: str, manifest_path:str):
    import importlib.util
    import kfp
    import sys

    spec = importlib.util.spec_from_file_location("module.name", pipeline_path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    # see https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path/50395128#50395128

    kfp.compiler.Compiler().compile(foo.kfservingPipeline, manifest_path)

def main(prefix: str):
    print("prefix:", prefix)
    import os
    pipeline_path_list = lookup_pipeline_py(os.path.join(prefix, "pipelines"))
    for pipeline_path in pipeline_path_list:
        manifest_path = os.path.splitext(pipeline_path)[0]+'.yaml'
        dynamic_import_and_compile(pipeline_path, manifest_path)

import argparse
parser = argparse.ArgumentParser(
    prog='pipeline manifest generator',
    description='generate kubeflow pipeline manifest')
parser.add_argument("--appdir", help="app dir", dest="appdir",
                     default="./")
if __name__ == '__main__':
    args = parser.parse_args()
    main(args.appdir)
