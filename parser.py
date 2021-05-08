import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', type=int)
    parser.add_argument('-r', '--rules', type=int)
    parser.add_argument('-p', '--population', type=int)
    return parser
