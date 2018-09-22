
import pytoolkit as tk


def create_generator(mode, multiple_output=False):
    """generatorの作成。"""
    assert mode in ('bin', 'ss')
    kwargs = {}
    if multiple_output:
        kwargs['output_index'] = 1

    gen = tk.generator.Generator(multiple_input=True, multiple_output=multiple_output)

    if mode == 'bin':
        # binでcropは答えが変わっちゃう可能性があるのでやめとく
        gen.add(tk.image.RandomFlipLR(probability=0.5), input_index=0)
        gen.add(tk.image.RandomPadding(probability=0.25, mode='reflect'), input_index=0)
        gen.add(tk.image.RandomRotate(probability=0.25), input_index=0)
    else:
        gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True), input_index=0, **kwargs)
        gen.add(tk.image.Padding(probability=1, with_output=True), input_index=0, **kwargs)
        gen.add(tk.image.RandomRotate(probability=0.25, with_output=True), input_index=0, **kwargs)
        gen.add(tk.image.RandomCrop(probability=1, with_output=True), input_index=0, **kwargs)

    gen.add(tk.image.RandomAugmentors([
        tk.image.RandomBlur(probability=0.125),
        tk.image.RandomUnsharpMask(probability=0.125),
        tk.image.RandomBrightness(probability=0.25),
        tk.image.RandomContrast(probability=0.25),
    ], probability=0.125), input_index=0)

    if mode == 'bin':
        gen.add(tk.image.Resize((101, 101)), input_index=0)
    else:
        gen.add(tk.image.Resize((101, 101), with_output=True), input_index=0, **kwargs)

    return gen
