# Test
from text_anonymizer import TextAnonymizer
import time

'''
Example code to test TextAnonymizer
Also used in the performance testing.
'''

ITERATIONS = 10

# Init anonymizer to work in mask mode and two languages
text_anonymizer = TextAnonymizer(languages=['fi'])


text_fi = ('Nimet: Toivo, Sami, Seppo, Ahti, Veikko, Jaana, Tiina, Minna, Aura, Lumi, Virtanen, Salminen, Gröönroos, Suomi.' \
           '\nRekisterinumerot: ABC-123, CCC-111, DDD-222, CBA-321' \
           '\nSähköpostiosoitteeni: testi-veikko.nieminen@example.com, 28j30d2@example.com'  \
           '\nIP-Osoitteet 192.168.177.111, 127.0.0.1, 1.1.1.1. '
           '\nPuhelinnumerot: +358448888888, 044 888 8888, 044888888' \
           '\nHenkilötunnukset: 010130A100K, 020300-001G. '
           '\nIban testinumerot: FI49 5000 9420 0287 30, GB33BUKB20201555555555' \
           '\nOsoitteet: Meriharjuntie 1 A 1, 40100 Helsinki, Mannerheimintie 2' \
           '\nKiinteistötunnulset:  092-416-11-123, 999-999-12-44-M601. ' \
           '\nTiedostot: exceli.xlsx, dokkari.pdf. '
           '\nURL:t http://www.google.com., https://helsinki.fi'
           )

print("Anonymizer running...")
start_time = round(time.time() * 1000)
for i in range(ITERATIONS):
    anonymized_fi = text_anonymizer.anonymize_text(text_fi)
print(text_fi)
print("--")
print(anonymized_fi)
print(" ")
time_ms = round(time.time() * 1000)-start_time
print("{i} iterations took {ms}ms, avg {avg}ms".format(i=ITERATIONS, ms=time_ms, avg=time_ms/ITERATIONS))