# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Prints identifiers whose check digits are valid, for writers of test documents.

    python tool/valid_ids.py <kind> [count] [--seed N]

Kinds: nir (French social security), siren, siret, fr_iban, es_iban, gb_iban, dni, nie,
es_nss (Spanish social security), nino (UK national insurance), nhs (UK NHS number),
us_ssn, card (Luhn, 16 digits). Everything is random: no number belongs to anyone on purpose.
Standard algorithms only, written without looking at the detection engine.
"""
import argparse
import random
import string

DNI_LETTERS = 'TRWAGMYFPDXBNJZSQVHLCKE'


def luhn_digit(body):
    total = 0
    for i, ch in enumerate(reversed(body)):
        n = int(ch)
        if i % 2 == 0:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return str(-total % 10)


def digits(r, n):
    return ''.join(r.choice(string.digits) for _ in range(n))


def iban(country, bban):
    rearranged = ''.join(str(int(c, 36)) for c in bban + country + '00')
    return f'{country}{98 - int(rearranged) % 97:02d}{bban}'


def grouped(s, size=4):
    return ' '.join(s[i:i + size] for i in range(0, len(s), size))


def nir(r):
    body = f'{r.choice("12")}{r.randint(50, 99):02d}{r.randint(1, 12):02d}{r.choice(["13", "31", "33", "35", "44", "59", "67", "69", "75", "92"])}{r.randint(1, 450):03d}{r.randint(1, 900):03d}'
    key = 97 - int(body) % 97
    return f'{body[0]} {body[1:3]} {body[3:5]} {body[5:7]} {body[7:10]} {body[10:13]} {key:02d}'


def siren(r):
    body = digits(r, 8)
    return body + luhn_digit(body)


def siret(r):
    body = siren(r) + '000' + r.choice('1234')
    return body + luhn_digit(body)


def fr_rib_key(bank, branch, account):
    return 97 - (89 * int(bank) + 15 * int(branch) + 3 * int(account)) % 97


def fr_iban(r):
    bank, branch, account = digits(r, 5), digits(r, 5), digits(r, 11)
    return grouped(iban('FR', f'{bank}{branch}{account}{fr_rib_key(bank, branch, account):02d}'))


def es_iban(r):
    bank, branch, account = digits(r, 4), digits(r, 4), digits(r, 10)

    def dc(s, weights):
        k = 11 - sum(int(c) * w for c, w in zip(s, weights)) % 11
        return {10: '1', 11: '0'}.get(k, str(k))
    control = dc('00' + bank + branch, [1, 2, 4, 8, 5, 10, 9, 7, 3, 6]) + dc(account, [1, 2, 4, 8, 5, 10, 9, 7, 3, 6])
    return grouped(iban('ES', bank + branch + control + account))


def gb_iban(r):
    return grouped(iban('GB', r.choice(['NWBK', 'BARC', 'LOYD', 'HBUK', 'MIDL']) + digits(r, 6) + digits(r, 8)))


def dni(r):
    n = r.randint(10_000_000, 99_999_999)
    return f'{n}{DNI_LETTERS[n % 23]}'


def nie(r):
    prefix, n = r.choice('XYZ'), r.randint(1_000_000, 9_999_999)
    return f'{prefix}{n}{DNI_LETTERS[int(str("XYZ".index(prefix)) + str(n)) % 23]}'


def es_nss(r):
    province, number = f'{r.randint(1, 52):02d}', digits(r, 8)
    base = int(province + number) if int(number) >= 10_000_000 else int(province) * 10_000_000 + int(number)
    return f'{province}/{number}/{base % 97:02d}'


def nino(r):
    first = r.choice('ABCEGHJKLMNOPRSTWXYZ')
    second = r.choice('ABCEGHJKLMNPRSTWXYZ')
    while first + second in ('BG', 'GB', 'NK', 'KN', 'TN', 'NT', 'ZZ'):
        second = r.choice('ABCEGHJKLMNPRSTWXYZ')
    d = digits(r, 6)
    return f'{first}{second} {d[0:2]} {d[2:4]} {d[4:6]} {r.choice("ABCD")}'


def nhs(r):
    while True:
        body = '4' + digits(r, 8)
        check = 11 - sum(int(c) * w for c, w in zip(body, range(10, 1, -1))) % 11
        if check != 10:
            return f'{body[0:3]} {body[3:6]} {body[6:9]}{check % 11}'


def us_ssn(r):
    return f'{r.randint(100, 665):03d}-{r.randint(10, 99):02d}-{r.randint(1000, 9999):04d}'


def card(r):
    body = r.choice(['4', '51', '52', '55']) + digits(r, 14)
    body = body[:15]
    return grouped(body + luhn_digit(body))


KINDS = {f.__name__: f for f in (nir, siren, siret, fr_iban, es_iban, gb_iban, dni, nie, es_nss, nino, nhs, us_ssn, card)}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=sorted(KINDS))
    ap.add_argument('count', nargs='?', type=int, default=5)
    ap.add_argument('--seed', type=int)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    for _ in range(args.count):
        print(KINDS[args.kind](rnd))
