import re

import paths

MIN_WORDS = 12

import re

maj = "A-ZÁÉÍÓÚÜÑÀÈÌÒÙÊÇÂÏÖ"
mn = "a-záéíóúüñàèìòùêçâïö"


subs = str.maketrans(
    "σναιун‡ρбйϊ─–сœъΙμόø‚ΑŽ" + "⇒−΅Ώｭ―…" + "\xad\x0c",
    "óíáéóíáñáéúñññúúÉÁüoeaé" + "\n\n\n\n\n\n\n\n" + "  ",
    "#+$†οεητίςδΒ&\t_θ"
)

s_open_exc = re.compile(fr'\b[1ijíl](?=[{maj}])')
s_1I_l = re.compile(fr'([{mn}])[1I]([{mn}])')
s_num_word = re.compile(fr'\b(\d+)([^\W\d]+)\b')
s_word_num = re.compile(fr'\b([^\W\d]+)(\d+)\b')
s_middle_maj = re.compile(fr'\b([{maj}]?[{mn}]+)([{maj}][{mn}]+)\b')


def clean_text(text: str):
    text = text.translate(subs)
    for k, v in {'ψ': 'psi', 'ϕ': 'phi', 'π': 'pi', 'æ': 'ae', 'Æ': 'ae'}.items():
        text = text.replace(k, v)
    text = re.sub(s_open_exc, "¡", text)
    text = re.sub(s_1I_l, r"\1l\2", text)
    text = re.sub(s_num_word, r"\1 \2", text)
    text = re.sub(s_word_num, r"\1 \2", text)
    text = re.sub(s_middle_maj, r"\1 \2", text)
    return text


def clean(file):
    (paths.ft_prep / file.name).write_text(
        '\n'.join(
            clean_text(line.strip())
            for line in file.read_text().split('\n')
            if len(re.findall(r'\s+', line.strip())) >= MIN_WORDS - 1
        )
    )


def main():
    for f in paths.ft_train.iterdir():
        clean(f)


if __name__ == '__main__':
    main()
