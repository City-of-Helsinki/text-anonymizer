# These sentences and strings are used for testing recognizers. Use only generated texts, not real addresses etc.

test_phonenumbers = ['040-0001119', '09 310 11111', '09 31011111', '0931011111',
                     '+358 (0)9 310 11111', '+3589 31011111', '+358931011111',
                      '+358 41 9879876', '+358419879876']

test_phonenumbers_fi = ['040 0001119', '09 310 1111', '090 310 1111', '093 1011 111']

test_addresses = ['Muoniontie 181, 90000 Kalavankoski', 'PL 1 (Pohjoisesplanadi 11-13), 00099 HELSINGIN KAUPUNKI',
                    'Mannerheimintie 5 A 00100 Helsinki', 'Mannerheimintie 5 A\n00100 Helsinki',
                     'Poste Restante\n00880 Helsinki', 'VALSSIMYLLYNKATU 11', 'Insinöörinkatu 3B',
                     'Wavulinintien', 'Leväluhdantiellä'
                    ]
test_addresses_spacy = ['Hei, peruutan kaupunkilehden. Osoitteeni on: Muoniontie 181, 90000 Kalavankoski',
                        'Kirjeen voi osoittaa meille osoitteeseen: PL 1 (Pohjoisesplanadi 11-13), 00099 HELSINGIN KAUPUNKI',
                        'Osoitteessa Mannerheimintie 5 A 00100 Helsinki on talo.',
                        'Lisään vielä että osoitteessa Mannerheimintie 5 A\n00100 Helsinki rivin vaihdon kera sijaitsee talo.',
                        'Vastaa: Kalle Testinen Poste Restante\n00880 Helsinki',
                        'AURAUS EI OLE TULLUT OSOITTEESEEN VALSSIMYLLYNKATU 11 KOLMEEN VIIKKOON', 'Insinöörinkatu 3B',
                        'Wavulinintien risteyksen kuoppa on yhä paikkaamatta.',
                        'Leväluhdantiellä ei ole hiekotettu tänäkään vuonna.',
                        'Tästä osoitteesta pitäisi saada peitettyä numerot: liisankatu 12 A 101 00100 Helsinki'
                    ]
test_names_fi = ['Maija Mehiläinen', 'Silja Laine', 'Marja Mustikkamäki', 'Teppo Tikka', 'Virtanen',
                'Silja Heikkinen',  'Irmeli', 'Timo', 'Antti', 'Minna', 'Anu', 'Antero', 'Antti Mäki',
                'Keijo', 'Salomaa', 'Nataljan']
test_names_en = ['Zhao Lê', 'Andrew Smith', 'John Doe', 'Jane Kreutz', 'Mary Johnson', 'Mary Johnson-Smith', 'Jerome K. Jerome', 'Anna K. Jerome', 'Alfred Kirby']
test_register_number = ['ABA-303', 'ABA303', 'ABA 303', 'aba-303', 'fdr-361', 'KQC546', 'abc123']
test_property_identifier = ['22-22-4444-333', '1-1-1-1', '22-1-333-1', '1-22-1-333', '1-22-1-333-4444',
                            '333-333-4444-4444-4444', '1-1-1-1-1', '14141414141414',
                            '151515151515151', '1616161616161616', '17171717171717171', '181818181818181818',
                            '1919191919191919191',  '91-7-104-3', '91-13-449-26']
test_ssn = ['150320-', '080320A242K', '190801-686P', '140500A509A',  '150399+111N', '150327', '121212-XXXX',
            '311299-999A', '010101-000A']

test_street = ['Alppikatu 4', 'Ahjokuja 3', 'Ahmatie', 'Mannerheimintie']

test_propn = ['Heikki', 'Sanna', 'Elmeri', 'Elina', 'Lampinen', 'Kalevi Sorsa', 'Kalle Koistinen']
test_email = [' osoite@palvelin.fi', 'osoite@palvelin.ali.com', 'kaksi-osainen.nimi@palvelu.org', ' tunnus@palvelin.aliorg.org.']
test_iban = ['FI49 5000 9420 0287 30', 'FI4950009420028730']
test_filenames = ['raimon_raportti.xls', 'eskon_excel.xlsx', 'kallen_kuva.jpg', 'kertun_kuva.png', 'ollin_opiskelu.pdf',
                  'erkin_esitys.ppt', 'elinan_esitys.pptx', ]

bad_ssn = ['0441234567', '0451234567', '0461234567', '421399-999L', '561399+999K', '421399A999J']
bad_register_number = ['300-223']
bad_phonenumbers = ['23.10.2021', '2021', '23-10-2021', '00600', '30.13', '33101', '9.1.1966']
bad_propn = ['Herjaa', 'vieläkään', 'esim', 'Yst.',
            'https://www.hel.fi/helsinki/fi/kulttuuri-ja-vapaa-aika/liikunta/liikuntakurssit/ilmoittautumisohjeet/',
            'Pe21.1.22', 'cii-486']
bad_address = ['liikuntahallia', 'ainakaan', 'kello', 'kävelytie tukittu', 'vuotava',
            'hallintokulttuuria']
bad_email = ['@tagi']
bad_iban = []
bad_property_identifier = ['23.10.2021', '040-0001119', '09 31011111', '09 310 11111']
bad_filenames = ['xls-tiedosto', 'pdf-tiedosto']

test_natural_language = [
    'Kankaanpääntien risteyksessä on punainen talo.',
    'Seppo Hovi soitti iloiseesti haitaria.',
    'Kallion kaupunginosa on tunnettu kuppiloistaan.'
]


default_test_cases = ['Matti Mäkinen', 'FI49 5000 9420 0287 30', '050 1234 121']


all_test_strings = test_phonenumbers + test_addresses + test_names_fi + test_register_number + test_property_identifier + test_ssn
