
import pytoolkit as tk


def create_generator(mode):
    """generatorの作成。"""
    assert mode in ('bin', 'ss')

    gen = tk.generator.Generator(multiple_input=True)
    gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True), input_index=0)

    if mode == 'bin':
        # binでcropは答えが変わっちゃう可能性があるのでやめとく
        gen.add(tk.image.RandomPadding(probability=0.25, mode='reflect'), input_index=0)
        gen.add(tk.image.RandomRotate(probability=0.25), input_index=0)
    else:
        gen.add(tk.image.Padding(probability=1, with_output=True), input_index=0)
        gen.add(tk.image.RandomRotate(probability=0.25, with_output=True), input_index=0)
        gen.add(tk.image.RandomCrop(probability=1, with_output=True), input_index=0)

    gen.add(tk.image.RandomAugmentors([
        tk.image.RandomBlur(probability=0.125),
        tk.image.RandomUnsharpMask(probability=0.125),
        tk.image.RandomBrightness(probability=0.25),
        tk.image.RandomContrast(probability=0.25),
    ], probability=0.125), input_index=0)

    gen.add(tk.image.Resize((101, 101)), input_index=0)

    if mode == 'ss':
        gen.add(tk.generator.ProcessOutput(lambda y: tk.ndimage.resize(y, 101, 101)))

    return gen
