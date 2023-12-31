import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-mc', '--model_ckpt', type=str, default='last_ckpt.pth')
    args = parser.parse_args()
    return args


def main():
    args = vars(get_args())
    print(args)

    module = __import__('diffae', fromlist=['DiffusionAutoEncodersInterface'])
    interface_cls = getattr(module, 'DiffusionAutoEncodersInterface')
    interface = interface_cls(args, mode='test')

    interface.test()


if __name__ == '__main__':
    main()
