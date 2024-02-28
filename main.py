import argparse
import sys, os
from utils.torchlight import import_class


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    processors = dict()

    # Trian the ActFormer generator or generate fake samples with the trained ActFormer
    processors['generator'] = import_class('processor.generator.GEN_Processor')

    # Train the action recognition model for evaluation
    processors['recognition'] = import_class('processor.recognition.REC_Processor')

    # Evaluate synthetic samples with the trained action recognition model
    processors['evaluator'] = import_class('processor.evaluator.Score_Processor')

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])


    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])
    os.makedirs(p.arg.work_dir, exist_ok=True)

    p.start()