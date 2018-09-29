
import pytoolkit as tk


def create_generator(mode, multiple_input=True, multiple_output=False):
    """generatorの作成。"""
    assert mode in ('bin', 'ss')
    kwargs = {}
    if multiple_input:
        kwargs['input_index'] = 0
    if multiple_output:
        kwargs['output_index'] = 1

    gen = tk.generator.Generator(multiple_input=multiple_input, multiple_output=multiple_output)

    if mode == 'bin':
        # binでcropは答えが変わっちゃう可能性があるのでやめとく
        gen.add(tk.image.RandomFlipLR(probability=0.5), **kwargs)
        gen.add(tk.image.RandomPadding(probability=0.25, mode='reflect'), **kwargs)
        gen.add(tk.image.RandomRotate(probability=0.25), **kwargs)
    else:
        gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True), **kwargs)
        gen.add(tk.image.Padding(probability=1, with_output=True), **kwargs)
        gen.add(tk.image.RandomRotate(probability=0.25, with_output=True), **kwargs)
        gen.add(tk.image.RandomCrop(probability=1, with_output=True), **kwargs)

    gen.add(tk.image.RandomAugmentors([
        tk.image.RandomBlur(probability=0.125),
        tk.image.RandomUnsharpMask(probability=0.125),
        tk.image.RandomBrightness(probability=0.25),
        tk.image.RandomContrast(probability=0.25),
    ], probability=0.125), **kwargs)

    gen.add(tk.image.Resize((101, 101), with_output=mode != 'bin'), **kwargs)

    return gen
