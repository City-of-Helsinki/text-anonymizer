import datetime

from spacy import load
from spacy.training import Example
from tabulate import tabulate

'''
This script evaluates the trained model with a set of sentences.
The evaluation data consists of sentences with single entity in each sentence.
The entity type is provided in the evaluation data.
The script evaluates the model with the evaluation data and prints the results.
'''

def evaluate_nlp(nlp=None):
    if not nlp:
        # Load trained model
        model_path = "../custom_spacy_model/fi_datahel_spacy-0.0.2"
        nlp = load(model_path)

    # Entity types
    LOC = 'LOC'
    AREA_ENTITY = 'GPE'
    PERSON = 'PERSON'
    O = 'O'
    CARDINAL = 'CARDINAL'
    DATE = 'DATE'
    ORG = 'ORG'
    ORDINAL = 'ORDINAL'
    GPE = 'GPE'


    # Evaluation data
    # We use these sentences to evaluate trained model.
    # Some are generated with large language model and some are quoted from wikipedia
    # Feel free to add sentences, but make sure that there is single entity in each sentence
    # If script fails, it is likely that spacy already detects that entity as different type or it is in different position.
    # Eg. 2022 -> Vuonna 2022

    sentence_tuples = [
        # Generated feedback simulation
        ("Kaupungin puutarhuri {} ilmaisi viime viikon lehdessä, että kaupungin viheralueiden hoito on parantunut merkittävästi.", "Liinus Järvenpää", PERSON),
        ("Kiitos, edustaja {}, uudesta pyörätiestä - se on parantanut liikkumistani kaupungissa.", "Gideon Lehti", PERSON),
        ("Toivon, että kaupunginvaltuutettu {} ottaa huomioon meidän alueen koulujen tarpeet tulevassa budjetissa.", "Romeo Majuri", PERSON),
        ("{} kirjoitti lehdessä, että kaupungin puistojen siisteys on parantunut huomattavasti viime kuukausina.", "Pirjo Siekkinen", PERSON),
        ("Tilaisuuden aluksi {} kiitti kaupungin työntekijöitä erinomaisesta työstä kaupungin siisteyden parantamiseksi.", "Alan Paavilainen", PERSON),
        ("{} ehdotti kaupunginvaltuustossa, että kaupunki investoisi lisää resursseja julkisten tilojen siisteyteen.", "Sven Karlsson", PERSON),
        ("{} 10:n kohdalla on suuri kuoppa tiessä, joka vaatii korjausta.", "Mannerheimintie", LOC),
        ("Havaittu vakava vaurio {} 15:n kohdalla, joka voi aiheuttaa vaaratilanteita.", "Aleksanterinkatu", LOC),
        ("{} 5:n edessä oleva tie on erittäin huonossa kunnossa ja tarvitsee pikaisen korjauksen.", "Erottajankatu", LOC),
        ("Olen erittäin tyytymätön keskusta-aluueen talvikunnossapitoon, erityisesti {} osalta.", "mannerheimintien", LOC),
        ("Kaupungininsinööri {} ansaitsee kiitoksen panoksestaan kaupungin it-koulutusjärjestelmän kehittämisessä.", "Akusti Kesti", PERSON),
        ("Toivon, että {} kunnossapitoon panostettaisiin enemmän, sillä liikennemäärät ovat kasvussa.", "pohjoisesplanadin", LOC),
        ("Kiitos kaupungin kesätyöntekijöille, jotka ovat pitäneet {} paikat siistinä ja istutukset kauniina.", "fredrikinkadulla", LOC),
        ("Tiedoksi että {} 2 kohdalla katuvalo on ollut pimeänä jo usean viikon ajan.", "kalevankatu", LOC),
        ("Kaupunginvaltuutettu {} lupasi vaalikampanjassaan panostaa kaupungin puistojen viihtyisyyteen ja siisteyteen.", "Sisu Asikainen", PERSON),
        ("Pyydän kiinnittämään huomiota {} kadun liikennejärjestelyihin, jotka aiheuttavat turvallisuusriskejä.", "mechelininkadun", LOC),
        ("Haluaisin kiittää lakaisukoneen kuljettajia, jotka ovat tänä keväänä huolehtineet {} kunnossapidosta erinomaisella tavalla.", "kaisaniemenkadun", LOC),
        ("Toivon, että kaupunki panostaisi enemmän resursseja {} valaistuksen parantamiseen.", "bulevardin", LOC),
        ("{} esiintyi viime viikolla kaupunki ja kirja tapahtumassa.", "Jukka-Pekka Halonen", PERSON),
        ("Kiitos {}, että olet kuunnellut asukkaiden toiveita ja parantanut kaupungin viheralueiden hoitoa.", "Vilppu Meriläinen", PERSON),
        ("En voi käsittää miksi {} on niin huonossa kunnossa, että se aiheuttaa jo vahinkoja autoille.", "unioninkatu", LOC),
        ("Kaupungin työntekijät ovat tehneet loistavaa työtä {} kadun talvikunnossapidossa.", "tehtaankadun", LOC),
        ("Toivoisin että {} pyöräilyolosuhteita parannettaisiin, jotta turvallinen liikkuminen olisi mahdollista.","hämeentien", LOC),
        ("Haluaisin kiittää kaupunginvaltuutettu {} siitä, että hän on nostanut esiin alueemme lähiökoulujen tarpeet.", "Tristan Kelaa", PERSON),
        ("Valitettavasti {} roskakorit ovat jatkuvasti täynnä, mikä aiheuttaa haittaa asukkaille.", "Yrjölänkujan", LOC),
        ("Edustaja {} lupasi vaalipuheessaan panostaa kaupungin koulutusjärjestelmän laadun parantamiseen.", "Räty", PERSON),
        ("Toivon, että kaupunginvaltuutettu {} ottaisi vakavasti asukkaiden huolenaiheet alikulkujen turvallisuudesta.","Olli-Pekka Heinonen", PERSON),
        ("Kiitos kaupungin työntekijöille, jotka ovat pitäneet {} kadun puhtaina ja hoidettuina.", "simonkadun", LOC),
        ("Olemme erittäin pettyneitä siihen, että {} jalkakäytävä on jatkuvasti huonossa kunnossa.", "soidinkujan", LOC),
        ("Toivoisin että {} pyöräilyolosuhteita parannettaisiin, jotta turvallinen liikkuminen olisi mahdollista.", "Kaisaniemenkadun", LOC),
        # Wikipedia quotes with names or street names
        ("Kampus sijaitsee Helsingin ydinkeskustassa Kruununhaan ja Kluuvin kaupunginosissa {} molemmin puoli", "Unioninkadun", LOC),
        ("Vuonna 1869 vihittiin Senaatintorin vastakkaisella puolella käyttöön mahtava kemian laboratorio- ja museorakennus Arppeanum, jonka {} oli suunnitellut venetsialaiseen tyyliin.", "Carl Albert Edelfelt", PERSON),
        ("Ensimmäinen Tiedekulma avattiin vuonna 2012 yliopiston hallintorakennukseen osoitteeseen {} 7", "Aleksanterinkatu", LOC),
        ("Helsingin yliopiston päärakennus sijaitsee Senaatintorin laidalla osoitteessa {} 1", "Unioninkatu", LOC),
        ("Kansalliskirjasto on osa Helsingin yliopistoa ja sijaitsee Senaatintorin laidalla osoitteessa {} 1", "Fabianinkatu", LOC),
        ("Ajatuksen seuran perustamisesta sai {}.", "Franz Fredrik Wathén", PERSON),
        ("Jalkapallossa HJK palasi Suomen-mestariksi valmentaja {} johdolla vuonna 1964.", "Aulis Rytkösen",  PERSON),
        ("HJK:n logo on peräisin vuodelta 1913 ja sen on suunnitellut {}.", "Osmo Korvenkontio", PERSON),
        ("{} on yksi kaupungin tärkemmistä ostoskaduista.", "Aleksanterinkatu", LOC),
        ("Helsingin kantakaupungin vanhin säilynyt rakennus on 1757 valmistunut talo {} risteyksessä", "Katariinankadun", LOC),
        ("{} piirtämiä rakennuksia on eritoten Senaatintorin ympärillä", "Engelin", PERSON),
        ("Tällaista suuntausta edustaa myös {} keskustasuunnitelmasta ainoana toteutettu Finlandia-talo.", "Alvar Aallon", PERSON),
        ("Helsingin suurin säännöllinen urheilutapahtuma on vuodesta 1976 alkaen järjestetty lasten ja nuorten {} Helsinki Cup", "Jalkapalloturnaus", O),
        ("Jääkiekossa Helsingin suosituimmat ja menestyneimmät joukkueet ovat SM-liigajoukkue {} sekä nykyisin Mestiksessä pelaava Jokerit, joka vuosina 2014–2022 pelasi itäeurooppalaisessa KHL-liigassa.", "HIFK", O),
        ("Helsingissä on kymmeniä elokuvasaleja, joista suurin on 635-paikkainen Tennispalatsin {} sali.", "ISENSE", O),
        ("Veneilijöitä varten kaupungin rannoilla on {} laituripaikkaa.", "noin 12 000", CARDINAL),
        ("Helsingin Satama on merkittävä yleisen liikenteen tuonti- ja vientisatama ja Suomen vilkkain matkustajasatama sekä risteily- että linjaliikenteessä. {} Helsingin satamassa oli 8 779 aluskäyntiä eli keskimäärin yli 24 laivaa joka päivä.", "Vuonna 2011", DATE),
        ("Suositun romaani 'Karhujen kuningas' kirjoittajaksi on mainittu {}, mutta kirjan todellinen tekijä on edelleen mysteeri.", "Gideon Timonen", PERSON),
        ("Fotografiskan uuden valokuvanäyttelyn avajaispuheen piti kuuluisa valokuvaaja {}, joka on tunnettu omalaatuisista mustavalkokuvistaan.", "Nuutti Ek", PERSON),
        ("Vuoden nuori yrittäjä -palkinto myönnettiin tänä vuonna innovatiivisesta teknologiastartupistaan {}.", "Pessi Jalkaselle", PERSON),
        ("Viimeisin suomalainen Nobelin palkinnon saaja {}, on tunnustettu hänen tutkimuksistaan kvanttifysiikan parissa.", "Altti Ekholm", PERSON),
        ("Kuuluisa säveltäjä {}, joka tunnetaan myös kiehtovasta elämäkerrastaan, johti eilen konsertin avajaisseremoniaa.", "Viking Hujanen", PERSON),
        ("Kirjailija {} on tehnyt läpimurron modernin kirjallisuuden kentällä hänen viimeisimmällä teoksellaan.", "Lukas Hautala", PERSON),
        ("Helsingissäkin tunnettu oopperalaulaja {} debytoi La Scalassa ja sai kriitikoilta loistavia arvosteluja.", "Ensio Tarkiainen", PERSON),
        ("Dokumentaristi {} esittelee tuoreen elokuvansa, joka keskittyy ilmastonmuutoksen vaikutuksiin arktisilla alueilla.", "Kerkko Lappi", PERSON),
        ("Uuden poliisijohtajan valinnassa eturiviin on noussut komisario {}, jonka ansioluetteloon kuuluu monia merkittäviä rikostutkintoja.", "Maxim Vainio", PERSON),
        ("Maailmankuulu kapellimestari {} johti eilen ilmiömäisesti Helsingin kaupunginorkesteria.", "Viking Rahkonen", PERSON),
        ("Modernin taiteen museon uusi kuraattori, {}, on jo aloittanut työnsä ja lupailee tuovansa näyttelyihin kansainvälisiä suuruuksia.", "Gideon Lehti", PERSON),
        ("Mysteerinovelli 'Synkkä lammen syvyys' on ylittänyt myyntiennätyksiä sen jännittävän juonen ja mestarillisesti kirjoitetun käsikirjoituksen, jonka takaa löytyy kirjailija {}.", "Romeo Majuri", PERSON),
        ("Maastopyöräilykisan voiton vei tällä kertaa ylivoimaisesti nuori ja lahjakas urheilija {}, joka on aiemminkin palkittu kyvyistään kovissa kilpailuissa.", "Sisu Asikainen", PERSON),
        ("Arkkitehti {}, joka on suunnitellut useita ekologisia asuintaloja, palkittiin viimeisimmästä projektistaan kestävän kehityksen messuilla.", "Lukas Koljonen", PERSON),
        ("Klassisen kitaran soittaja {} lumosi yleisön intensiivisellä tulkinnallaan Bachin teoksista viimeisimmällä Euroopan-kiertueellaan.", "Viking Leskinen", PERSON),
        ("Maajussi-kilpailun voitti yllättäen {}, jonka lämminhenkiset ja ahkerat tavat voittivat sekä tuomariston että katsojien sydämet.", "Akusti Ikävalko", PERSON),
        ("Komediasarjan pääosaa näyttelevä {} on noussut uudeksi fanisuosikiksi hauskalla otteellaan ja luontevalla roolisuorituksellaan.", "Ylermi Toiviainen", PERSON),
        ("Tohtori {} esitteli eilen läpimurtotutkimustaan neurotieteiden konferenssissa, ja hänen työnsä saa varmasti jatkossakin paljon huomiota.", "Gideon Olli", PERSON),
        ("Reservin upseerikerhon vuosipäivää juhlisti puheenvuorollaan eversti {}, joka muisteli uransa merkittävimpiä hetkiä ja tapahtumia.", "Maxim Tiitinen", PERSON),
        ("Avantgardistinen koreografi {} on tuonut tanssiesitykseen mullistavan tavan käyttää modernia teknologiaa ja perinteistä balettia upeasti yhdistäen.", "Lukas Ström", PERSON),
        # Generated using wikipedia
        ("Helsingissä toimii {} seudun liikenteen (HSL) järjestämänä kattava julkinen liikenne, joka koostuu linja-autojen säteittäis- ja poikittaisyhteyksistä, kymmenen linjan raitiotiejärjestelmästä, Espooseen ulottuvasta kaksihaaraisesta metroradasta sekä kolmesta lähijunilla liikennöitävästä kaupunkiradasta.", "Helsingin", GPE),
        ("Lentoliikennettä palvelee Vantaalla sijaitseva {} lentoasema, jonka alle rakennettu Lentoaseman rautatieasema avattiin osana kehärataa 10. heinäkuuta 2015.", "Helsinki-Vantaan", GPE),
        ("Vuonna 1968 {} hotelleissa majoittui noin 328 000 henkeä, joista runsaat 130 000 oli ulkomaalaisia.", "Helsingin", GPE),
        ("Vuoteen 2021 asti yleisilmailua varten käytössä oli pienempi {} lentoasema.", "Helsinki-Malmin", GPE),
        ("Vuonna 2011 {} satamassa oli 8 779 aluskäyntiä eli keskimäärin yli 24 laivaa joka päivä.", "Helsingin", GPE),
        ("Helsingin päärautatieasema on Suomen matkustajaliikenteen keskus. Noin kolme kilometriä pohjoisemmalla {} rautatieasemalla rautatie haarautuu rantaratana länteen ja pääratana pohjoiseen.", "Pasilan", LOC),
        ("Eräs merkittävä säteittäinen väylä on {}", "Itäväylä", LOC),
        ("Hotelliyöpymiset Helsingissä ovat lähes 13-kertaistuneet 50 vuodessa, ulkomaalaisten osalta lähes 18-kertaistuneet. Vuonna 2018 {} hotelleissa yöpyi lähes 4,2 miljoonaa henkeä, joista 2,3 miljoonaa ulkomaalaista.", "Helsingin", GPE),
        ("Veneilijöitä varten kaupungin rannoilla on {} laituripaikkaa.", "noin 12 000", CARDINAL),
        ("Helsingin Satama on merkittävä yleisen liikenteen tuonti- ja vientisatama ja Suomen vilkkain matkustajasatama sekä risteily- että linjaliikenteessä. {} Helsingin satamassa oli 8 779 aluskäyntiä eli keskimäärin yli 24 laivaa joka päivä.", "Vuonna 2011", DATE),
        ("Kalasatama (ruots. Fiskehamnen[4]) on {} kaupunginosan osa-alue Helsingin itäisessä kantakaupungissa.", "Sörnäisten", LOC),
        ("Se käsittää suuren osan entistä {} sataman aluetta, jolle on alettu rakentaa uutta laajaa asunto- ja toimistoaluetta.", "Sörnäisten", LOC),
        ("Kalasataman keskuksen alueelle metroaseman ympärille rakentuu 23–35-kerroksisten tornitalojen keskittymä, joista Majakka valmistui {}, Loisto syyskuussa 2021, Lumo One elokuussa 2022 ja Visio joulukuussa 2023.", "vuonna 2019", DATE),
        ("Tornitalojen keskiössä on {} avattu kauppakeskus Redi.", "syyskuussa 2018", DATE),
        ("Liikenteellisesti Kalasatama on keskeisellä paikalla, ja {} kaasukellojen pohjoispuolella sijaitseva pääkatujen ja Seututie 170:n muodostama liittymäalue on kantakaupungin merkittävimpiä solmupisteitä.", "Suvilahden", LOC),
        ("Alueelle valmistui Kalasataman metroasema {}.", "vuonna 2007", DATE),
        ("{} aseman itäinen sisäänkäynti yhdistettiin vastavalmistuneeseen Rediin, mikä mahdollisti suoran pääsyn metroasemalta kauppakeskukseen ja sen päällä oleviin tornitaloihin.", "Syyskuussa 2018", DATE),
        ("Alueelle rakennetaan myös uutta raitiotieyhteyttä {}.", "Pasilaan", LOC),
        ("Kalasatamasta on tulossa varsin tiiviisti rakennettu alue – asukkaita sinne arvioidaan tulevan lopulta {}, suunnilleen yhtä paljon kuin Kalliossa.", "noin 20 000", CARDINAL),
        ("Lisäksi alueelle kaavaillaan työpaikkoja {} ihmiselle.", "noin 10 000", CARDINAL),
        ("Saaren nimi Byholmen oli käytössä jo 1600-luvulla. P. Klerckin kartassa {} saaren nimi on Kråkholmen (Varissaari).", "vuodelta 1776", DATE),
        ("Kuninkaallisessa merikartastossa {} nimi on muodossa Krok-Holmen (Koukkusaari).", "1791–1796", DATE),
        ("P. S. von Schrarenbergin Helsingin ja Sipoon kartassa {} saaren nimi on Stenholmen (Kivisaari).", "vuodelta 1835", DATE),
        ("Sitä kutsuttiin {} myös Hemholmeniksi (suom. Kotisaari), mutta vuosisadan lopulla nimeksi vakiintui jälleen Byholmen.", "1800-luvulla", DATE),
        ("Suomenkielinen nimi Kyläsaari vahvistettiin {}.", "vuonna 1909", DATE),
        ("Läheisestä Toukolan kaupunginosasta löytyy {}, jossa sijainnut Kotisaaren leipomo, entinen Maanviljelijöiden maitokeskus, on saanut nimensä tästä paikannimestä.", "Kotisaarenkatu", LOC),
        ("{} Kyläsaareen valmistui jätevedenpuhdistamo,", "Vuonna 1932", DATE),
        ("{} saari liitettiin mantereeseen.", "1940-luvulla", DATE),
        ("1960-luvun alussa alueelle rakennettiin Kyläsaaren jätteenpolttolaitos, joka kuitenkin lakkautettiin {}", "vuonna 1983", DATE),
        ("Pääkaupunkiseudun Kierrätyskeskus aloitti toimintansa jätteenpolttolaitoksen rakennuksessa {}.", "vuonna 1990", DATE),
        ("Diakonia-ammattikorkeakoulun eli Diakin rakennus valmistui osoitteeseen {} 2 vuonna 2015.", "Kyläsaarenkuja", LOC),
        ("Kyläsaaren ranta-alueella on osittain kuvattu {} elokuva Mies vailla menneisyyttä.", "Aki Kaurismäen", PERSON),
        ("Kyläsaari esiintyy myös {} romaanissa Rakastunut rampa, vuodelta 1922.", "Joel Lehtosen", PERSON),
        ("Siinä kuvataan juhannuksen viettoa luonnonkauniissa {}.", "Kyläsaaressa", GPE)
    ]


    # Build examples from eval set
    all_eval_data = []
    for sentence, entity_value, entity_label in sentence_tuples:
        # Build entities
        start = sentence.index("{")
        end = start + len(entity_value)
        entities = [[start, end, entity_label]]
        # Setup example dict
        annotations = {"entities": entities}
        # Create example object
        example = Example.from_dict(nlp.make_doc(sentence.format(entity_value)), annotations)
        all_eval_data.append(example)


    #
    # Evaluation results
    #

    eval_results = nlp.evaluate(all_eval_data)
    eval_results_data = eval_results['ents_per_type']

    #
    # Info about the dataset
    #
    entity_counts = {}
    # Iterate over the evaluation data
    for example in all_eval_data:
        # Access the entities
        for entity in example.reference.ents:
            # Increment the count of the entity type
            entity_counts[entity.label_] = entity_counts.get(entity.label_, 0) + 1

    # Combine the evaluation results and entity counts into markdown tables
    for key, value in eval_results_data.items():
        for key2, value2 in entity_counts.items():
            if key == key2:
                value['count'] = value2

    # Convert the dictionary to a list of lists
    eval_results_table_data = [[key] + list(values.values()) for key, values in eval_results_data.items()]
    # Define the headers
    headers = ['Entity', 'precision', 'recall', 'f1-score', 'samples']

    # Create the markdown table for evaluation results
    eval_results_markdown_table = tabulate(eval_results_table_data, headers, tablefmt="pipe")

    #
    # Print results
    #
    print("\n\n### Evaluation results for model")
    print("\nEvaluation dataset consists of {} sample sentences.\n".format(len(all_eval_data)))

    print(f"\nDate: { datetime.datetime.now().strftime('%d.%m.%Y')}\n")
    print("\nEvaluation results: \n")
    print(eval_results_markdown_table)
    print("")
    return eval_results_markdown_table


if __name__ == "__main__":
    evaluate_nlp()
