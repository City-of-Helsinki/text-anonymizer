import csv
import datetime
import itertools
import os
import random

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

from evaluation import evaluate_nlp

print("Starting fine tuning of spacy model for Finnish names, helsinki streets and areas")

AREAS_TEST_DATA_SIZE = 175
STREETS_TEST_DATA_SIZE = 975
NAMES_TEST_DATA_SIZE = 675


exec_ner = True
exec_test = True
exec_ruler = True
save_model = True

# Use fixed seed so training will always be the same
random.seed(1234)

STREET_ENTITY = 'LOC'
AREA_ENTITY = 'GPE'
NAME_ENTITY = 'PERSON'

base_model = "fi_core_news_lg"
nlp = spacy.load(base_model)
target_path = "../custom_spacy_model/fi_datahel_spacy-0.0.2"

this_dir, this_filename = os.path.split(__file__)

_FIRST_NAMES_FILE_PATH = "../test/data/etunimet.csv"
_FIRST_NAMES_DATA_FILE = os.path.join(this_dir, _FIRST_NAMES_FILE_PATH)
_FIRST_NAMES = []

_LAST_NAMES_FILE_PATH = "../test/data/sukunimet.csv"
_LAST_NAMES_DATA_FILE = os.path.join(this_dir, _LAST_NAMES_FILE_PATH)
_LAST_NAMES = []

_STREETS_FILE_PATH = "../test/data/helsinki_kadunnimet.txt"
_STREETS_DATA_FILE = os.path.join(this_dir, _STREETS_FILE_PATH)
_STREETS = []

_AREAS_FILE_PATH = "../test/data/helsinki_alueet.txt"
_AREAS_DATA_FILE = os.path.join(this_dir, _AREAS_FILE_PATH)
_AREAS = []

_PRODUCTS_FILE_PATH = "../test/data/tuotenimet.txt"
_PRODUCTS_DATA_FILE = os.path.join(this_dir, _PRODUCTS_FILE_PATH)
_PRODUCTS = []

_ORGANIZATIONS_FILE_PATH = "../test/data/organisaatiot.txt"
_ORGANIZATIONS_DATA_FILE = os.path.join(this_dir, _ORGANIZATIONS_FILE_PATH)
_ORGANIZATIONS = []

_SKIP_FILE_PATH = "../test/data/ohitettavat.txt"
_SKIP_DATA_FILE = os.path.join(this_dir, _ORGANIZATIONS_FILE_PATH)
_SKIP = []

with open(_LAST_NAMES_DATA_FILE, 'r') as data:
    for line in csv.reader(data, delimiter=';'):
        _LAST_NAMES.append(line[0])
        # take top 2000 last names
        if len(_LAST_NAMES) >= 2000:
            break

with open(_FIRST_NAMES_DATA_FILE, 'r') as data:
    for line in csv.reader(data, delimiter=';'):
        _FIRST_NAMES.append(line[0] if random.randint(1, 3) > 1 else line[0].lower())
        # take top 2000 first names
        if len(_FIRST_NAMES) >= 2000:
            break

with open(_STREETS_DATA_FILE, 'r') as data:
    for line in csv.reader(data, delimiter=';'):
        _STREETS.append(line[0] if random.randint(1, 3) == 1 else line[0].lower())
        # take top 1000 street names
        # if len(_STREETS) >= 1000:
        #     break

with open(_AREAS_DATA_FILE, 'r') as data:
    for line in csv.reader(data, delimiter=';'):
        _AREAS.append(line[0] if random.randint(1, 3) == 1 else line[0].lower())
        # take top 200 area names
        # if len(_AREAS) >= 200:
        #     break

with open(_ORGANIZATIONS_FILE_PATH, 'r') as data:
    for line in csv.reader(data, delimiter=';'):
        _ORGANIZATIONS.append(line[0] if random.randint(1, 3) == 1 else line[0].lower())

with open(_PRODUCTS_DATA_FILE, 'r') as data:
    for line in csv.reader(data, delimiter=';'):
        _PRODUCTS.append(line[0] if random.randint(1, 3) == 1 else line[0].lower())


def generate_sentence(person, list):
    try:
        sent = random.choice(list)
        if '{s}' in sent:
            sent = sent.replace('{s}', person)
        if '{adj}' in sent:
            sent = sent.replace('{adj}', random.choice(ADJECTIVES))
        if '{adv}' in sent:
            sent = sent.replace('{adv}', random.choice(ADVERBS))

        s = sent.index(person)
        e = s + len(person)
        return sent, s, e
    except ValueError:
        print("Error: ", sent, person)
        return generate_sentence(person, list)

def generate_evaluation_sentence(value, sentence):
    sent = sentence.format(s=value)
    s = sent.index(value)
    e = s + len(value)
    return sent, s, e


def generate_full_names(amount=1):
    full_names = []
    for a in range(amount):
        random_first_name = random.choice(_FIRST_NAMES)
        # Add 2 first names sometimes
        if random.randint(1, 50) == 1:
            random_first_name += ' ' + random.choice(_FIRST_NAMES)
        elif random.randint(1, 50) == 2:
            random_first_name += '-' + random.choice(_FIRST_NAMES)


        # two part names
        random_last_name = random.choice(_LAST_NAMES)
        if random.randint(1, 50) == 3:
            random_last_name += '-' + random.choice(_LAST_NAMES)

        if random.randint(1, 50) == 4:
            random_name = random_last_name
        elif random.randint(1, 50) == 5:
            random_name = random_first_name
        else:
            random_name = random_first_name + ' ' + random_last_name
        full_names.append(random_name)
    return full_names


def test_text() -> bool:
    amount = 2
    names = generate_full_names(amount)
    names_flattened = list(itertools.chain.from_iterable([a.split(' ') for a in names]))

    # test_text = build_random_sentence(names)
    test_text = "Tämä on keksitty lause jolla testataan miten hyvin erilaiset nimet tunnistetaan anonymisoitavaksi. " \
                "Ala-asteen opettaja {name2} antoi pojalle uuden kumin ja kynän. Tästä tuli kaikille hyvä mieli." \
                "Päivä paistaa ja linnut laulaa, se on todella mukava asia! " \
                "Kiitos! Terkuin oppilaan vanhempi {name1}. ".format(name1=names[0], name2=names[1])
    doc = nlp(test_text)
    correct_label = 0
    for ent in doc.ents:
        entity_str = str(ent).replace('\\.', '')
        if ent.label_ == NAME_ENTITY and entity_str in names:
            correct_label += 1
        elif ent.label_ == NAME_ENTITY and entity_str in names_flattened:
            correct_label += 0.5
        elif ent.label_ == NAME_ENTITY:
            # print("Incorrect: ", ent, ent.label_)
            pass
    return correct_label >= amount and amount < correct_label + 1

def build_random_sentence(names: list[str]) -> str:
    s1 = generate_sentence(names[0], EVALUATION_SENTENCES)
    s2 = generate_sentence(names[1], EVALUATION_SENTENCES)
    return s1[0] + " " + s2[0]

def test_areas() -> bool:
    amount = 2
    area1 = random.choice(_AREAS)
    area2 = random.choice(_AREAS)
    test_text = "Tämä on keksitty lause jolla testataan miten hyvin erilaiset nimet tunnistetaan anonymisoitavaksi. " \
                "{area1} on loistava alue! Ala-asteen opettaja antoi pojalle uuden kumin ja kynän. Tästä tuli kaikille hyvä mieli." \
                "Päivä paistaa ja linnut laulaa, se on todella mukava asia! " \
                "Kiitos! {area2} on mukava paikka asua. ".format(area1=area1, area2=area2)
    doc = nlp(test_text)
    correct_label = 0
    for ent in doc.ents:
        if str(ent.label_) in [AREA_ENTITY, STREET_ENTITY] and str(ent) in _AREAS:
            correct_label += 1
        elif str(ent) in _AREAS:
            print("Incorrect: ", ent, ent.label_)
    return correct_label == amount


def test_streets() -> bool:
    amount = 2
    street1 = random.choice(_STREETS)
    street2 = random.choice(_STREETS)
    test_text = "Tämä on keksitty lause jolla testataan miten hyvin erilaiset katujen nimet tunnistetaan anonymisoitavaksi. " \
                "Osoitteessa {street1} 17 A 1 on puu, joka tarvitsee apua. " \
                "Olipa hieno taideteos myös! Terveisin, asukas kadulta {street2}.".format(street1=street1,
                                                                                          street2=street2)
    doc = nlp(test_text)
    correct_label = 0
    for ent in doc.ents:
        if str(ent.label_) in [AREA_ENTITY, STREET_ENTITY, 'STREET'] and str(ent) in _STREETS:
            correct_label += 1
        elif str(ent) in _STREETS:
            print("Incorrect: ", ent, ent.label_)
    return correct_label == amount


def run_test(amount=50):
    results1 = []
    for i in range(amount):
        results1.append(test_text())
        results1.append(test_areas())
        results1.append(test_streets())
    p = results1.count(True) / (amount * 3) * 100
    print("Test coverage %", p)
    return p


def build_patterns(data, label):
    print(f"Build {len(data)} patterns for  {label}")
    patterns = []
    for s in data:
        patterns.append({'pattern': s, 'label': label})
    return patterns

ADVERBS = ['hyvin', 'mukavasti', 'tyylikkäästi', 'oudosti', 'pohdiskellen', 'tuttavallisesti']
ADJECTIVES = ['hieno', 'mukava', 'tyylikäs', 'outo', 'pohdiskeleva', 'tuttavallinen', 'kiva', 'hauska', 'kummallinen', 'mielenkiintoinen', 'kaunis']



print("Building test data for training")
# areas
print(f"Generating {AREAS_TEST_DATA_SIZE} sentences with areas")
print(f"Generating {STREETS_TEST_DATA_SIZE} sentences with streets")
print(f"Generating {NAMES_TEST_DATA_SIZE} sentences with names")

AREA_LIST = random.sample(_AREAS, AREAS_TEST_DATA_SIZE)
STREET_LIST = random.sample(_STREETS, STREETS_TEST_DATA_SIZE)
NAME_LIST = generate_full_names(NAMES_TEST_DATA_SIZE)
TRAIN_DATA = []
SENTENCES_NAME = [
    '{s} on hyvä tyyppi.',
    '{s} on suomalainen miehen etunimi.',
    '{s} vakiintui Suomessa nimimuotona jo keskiajalla.',
    'Terveisin {s}.',
    '{s}, Tiepolku 10 Helsinki.',
    'Lähettäjä: {s} <sahkoposti.osoite@palvelin.com>.',
    'Lähettäjä: {s}.',
    '{s} pitäisi palkita hyvästä työstä.',
    'Meidät otti vastaan {s}.',
    'Koulun rehtori {s} sanoi että opiskelkaa matematiikkaa.',
    'Kaupunginhallituksen puutarhakerhon puheenjohtaja {s} liikuttui puheensa aikana kertoessaan pihakuusesta.',
    '{s} ja Teppo esiintyvät ensi viikon tiistaina.',
    'Kiitos {s}, autoit meitä auton kanssa.',
    'Terveisin: {s}.',
    'Kuusikankaan terveyskeskuksen lääkäri {s} oli mukava.',
    'Lähettäjä: {s} <maili123@palvelu.com>.',
    'Vastaanottaja: {s} <hs82hs72gho@hel.fi>.',
    '{s} Isännöinti Oy.',
    'TMI {s}.',
    'Kaupunginhallituksen {s} ansaitsee palkankorotuksen.',
    'Kunnan virkamies {s} osaa hommansa hyvin.',
    'Puisto-osaston päällikkö {s} ohjeisti minua pysäköimään oikealle paikalle.',
    'Kiittäen {s}.',
    'Hei {s}!.',
    'TERVE {s}.',
    'Nimi: {s}.',
    'Asiakas: {s}.',
    'Tervehdys {s}.',
    'Lapsemme {s} on tänään myöhässä koulusta.',
    'Naapurini {s} yritti tavoitella kaupungin muuraria.',
    'Isäni {s} ei saanut yhteyttä katulamppujen korjausosastoon.',
    'Pormestari {s} voisi ostaa hatun.',
    'Eilen illalla {s} pelasti kisun puusta.',
    '{s} on kirjoittanut useita bestsellereitä viime vuosina.',
    '{s} voitti maratonin ylivoimaisella ajalla.',
    '{s} luovutti palkintonsa hyväntekeväisyyteen.',
    '{s} lauloi kansallislaulun stadionilla.',
    '{s} perusti uuden teknologiayrityksen viime kuussa.',
    '{s} opettaa matematiikkaa paikallisessa yläkoulussa.',
    'Konsertissa {s} soitti viulua lumoavasti.',
    '{s} pelasti monta elämää työskentelemällä palomiehenä.',
    '{s} kehitti uuden sovelluksen, joka helpottaa arkielämää.',
    '{s} on tunnettu taidemaalari ja hänen näyttelynsä ovat aina täynnä.',
    '{s} valittiin vuoden yrittäjäksi.',
    '{s} esitti uuden tutkimuksensa tiedekonferenssissa.',
    '{s} osallistui tv-visailuun ja voitti suuren summan rahaa.',
    '{s} järjesti upean hyväntekeväisyysillallisen.',
    '{s} johti onnistuneesti suurta markkinointikampanjaa.',
    '{s} oli pääosassa menestyselokuvassa.',
    '{s} opiskeli kymmenen vuotta ennen kuin sai tohtorin arvon.',
    '{s} loi uuden koodauskielen, joka mullisti alaa {adv}.',
    '{s} otti haasteen vastaan ja kiipesi Mount Everestille.',
    '{s} piti inspiroivan puheen valmistuneille.',
    '{s} avasi uuden ravintolan kaupungin sydämessä.',
    '{s} keksi uuden nopean testin sairauksien havaitsemiseksi.',
    '{s} voitti arvostetun palkinnon innovatiivisesta muotoilusta.',
    '{s} on innokas puutarhuri ja hänen puutarhansa on nähtävyys.',
    '{s} omistaa pienen kahvilan, joka on tunnettu herkullisista leivonnaisistaan.',
    '{s} otti vastuun kansainvälisestä hankkeesta.',
    '{s} opetti lapsille tärkeyden ympäristönsuojelusta.',
    '{s} kirjoitti kolumnin, joka sai paljon huomiota sosiaalisessa mediassa.',
    '{s} {adv} pelasi maalin jääkiekon MM-finaalissa.',
    '{s} puhui YK:ssa {adv} ilmastonmuutoksen vaikutuksista.',
    '{s} loi uuden liikkeen tanssimaailmaan.',
    '{s} paransi yrityksen myyntiä {adv}.',
    '{s} kerää varoja uuteen lastensairaalaan.',
    '{s} on kapellimestari, joka johtaa kuuluisaa orkesteria.',
    '{s} julkaisi artikkelin, joka muutti näkemyksiä tieteessä.',
    '{s} perusti {ajd} hyväntekeväisyysjärjestön.',
    '{s} suunnitteli kestävän energiaratkaisun kyläyhteisöön.',
    '{s} voitti pääpalkinnon valokuvakilpailussa.',
    '{s} toimi tuomarina kansallisessa ruoanlaittokilpailussa.',
    '{s} antoi äänensä animaatioelokuvan päähenkilölle.',
    '{s} aloitti kampanjan puhtaan veden saamiseksi kaikille.',
    '{s} nousi johtoon tärkeässä poliittisessa puolueessa.',
    'Koko {ajd} kaupunki puhui siitä, kuinka {s} oli pelastanut koulun pihalla loukkaantuneen linnun. ',
    'Paikallinen leipuri {s} voitti kansainvälisen leivontakilpailun herkullisilla resepteillään.',
    'Kun {s} soitti viimeistä nuottia, konserttisalin yleisö nousi seisaalleen aplodeeraamaan.',
    'Viime viikolla {s} sai tunnustusta innovaatiosta, joka voi pelastaa miljoonia elämiä.',
    'Vaikka {s} oli vasta aloittelija, hän voitti shakkiturnauksen kokeneita vastustajia vastaan.',
    'Astrofysiikan professori {s} esitti uuden teorian mustista aukoista, joka saattaa mullistaa käsityksemme universumista.',
    'Sairaalan päivystysosaston johtaja {s} kiitti henkilökuntaansa heidän uhrauksistaan pandemian aikana.',
    'Kadonnutta koiraa etsiessään {s} löysi sen ja palautti sen turvallisesti omistajalle.',
    'Taitoluistelija {s} esitti täydellisen ohjelman ja sai yleisön kyyneliin.',
    'Kun {s} astui lavalle, hän oli heti karismaattinen ja vangitsi yleisön huomion.',
    'Tänä vuonna {s} julkaistiin aikakauslehden "Vuoden Vaikuttaja" -listalle.',
    'Arkkitehti {s} suunnitteli kestävän ekokylän, joka on nyt {ajd} malliesimerkki vihreästä asumisesta.',
    'Jäätelökaupan omistaja {s} keksi uuden suosikkimaun, joka sai asiakkaat jonottamaan ovella.',
    'Huippukokki {s} kertoi salaisuuden täydellisen risoton valmistuksesta.',
    'Kaupunginjohtaja {s} avasi uuden puiston, joka on omistettu paikallisille sankareille.',
    'Kun {s} esitteli uuden teoksensa galleriassa, kaikki olivat vaikuttuneita hänen kyvystään ilmaista tunteita.',
    'Lentopallon valmentaja {s} johdatti joukkueensa voittoon uskomattomalla strategialla.',
    'Opettaja {s} sai tunnustuksen omistautumisestaan erityisopetuksen edistämiselle.',
    'Kun {s} päätti puheensa {adv}, kaikki tiesivät, että he olivat kuulleet jotain merkittävää.',
    'Koripalloilija {s} teki voittoheiton juuri pelin viime sekunneilla.',
    'Toimittaja {s} paljasti laajan korruptioskandaalin, joka järkytti koko maata.',
    'Tohtori {s} kehitti uuden hoitomenetelmän, joka on auttanut tuhansia potilaita.',
    'Majuri {s} sai kunniamerkin rohkeudestaan ja johtajuudestaan.',
    'Kemisti {s} sai patentin uudelle, ympäristöystävälliselle polttoaineelle.',
    'Taikuri {s} hämmästytti yleisöä uskomattomilla tempuillaan.',
    'Nuori aktivisti {s} puhui intohimoisesti ilmastomuutoksen vastaisessa mielenosoituksessa.',
    'Diplomaatti {s} neuvotteli rauhansopimuksen, joka lopetti pitkään jatkuneen konfliktin.',
    'Kun {s} ilmoitti pyrkivänsä presidentiksi, monet olivat innoissaan mahdollisista muutoksista.',
    'Kapteeni {s} ohjasi laivan turvaan myrskyn keskeltä pelastaen kaikkien matkustajien hengen.',
    'Stand-up-koomikko {s} sai koko salin nauramaan ainutlaatuisella huumorintajullaan.',
    'Sarjakuvataiteilija {s} sai kulttuuripalkinnon luovuudestaan ja innovatiivisuudestaan.',
    'Asiantuntija {s} kommentoi talousuutisia tarjoten syvällistä ymmärrystä nykyisestä markkinatilanteesta.',
    'Kun {s} suoritti vaarallisen temppun, katsojat hengittivät helpotuksesta onnistumisen jälkeen.',
    'Kirurgi {s} suoritti monimutkaisen operaation, joka pelasti potilaan elämän.',
    'Ohjaaja {s} voitti palkinnon parhaasta elokuvasta kansainvälisillä elokuvafestivaaleilla.',
    'Sukellusmestari {s} opasti aloittelijoita merenalaisen maailman ihmeisiin.',
    'Professori {s} julkaistiin useiden tieteellisten artikkelien ansiosta.',
    'Kuoronjohtaja {s} harjoitteli kuoronsa kanssa kuukausia, ja lopulta he voittivat kansainvälisen kilpailun.',
    'Taidekriitikko {s} kirjoitti vaikuttavan arvion uudesta näyttelystä.',
    'Biologi {s} löysi uuden lajin sademetsästä, joka saattaa auttaa lääketieteen kehityksessä.',
    'Lastenkirjailija {s} innosti lapsia lukemaan hauskalla ja opettavaisella kirjasarjallaan.',
    'Elokuvatähti {s} lahjoitti merkittävän summan hyväntekeväisyysjärjestölle.',
    'Säveltäjä {s} sai standing ovation uuden sinfoniansa ensiesityksessä.',
    'Fysiikan opettaja {s} selitti kvanttimekaniikkaa tavalla, joka sai kaikki oppilaat ymmärtämään aiheen.',
    'Sotilas {s} palasi kotiin pitkän ulkomaankomennuksen jälkeen sankarillisena.',
    'Muotisuunnittelija {s} julkaisi uuden malliston, joka on saanut kiitosta ympäri maailmaa.',
    'Urheilutoimittaja {s} kirjoitti syvällisen analyysin olympiakisojen tapahtumista.',
    'Puuseppä {s} valmisti kauniita käsintehtyjä huonekaluja, jotka kestävät sukupolvelta toiselle.',
    'Laulaja {s} julkaisi uuden albumin, joka nousi heti listaykköseksi.',
    'Kirjastonhoitaja {s} järjesti lastentapahtuman, joka rohkaisi nuoria lukemaan.',
    'Oikeustieteen professori {s} piti luennon, joka valaisi monimutkaisia lakiasioita.',
    'Tutkimusmatkailija {s} palasi retkeltään tuoden mukanaan ainutlaatuista tietoa harvinaisista kasvilajeista.',
    'Paleontologi {s} löysi uuden dinosauruksen fossiilin, joka täydentää historiaamme.',
    'Lahjakas nuori muusikko {s} voitti arvostetun palkinnon kansainvälisessä kilpailussa.',
    'Ballerina {s} tanssi pääroolin baletissa, joka keräsi kiitosta kriitikoilta.',
    'Dokumentaristi {s} julkaisi vaikuttavan elokuvan, joka herätti keskustelua tärkeästä aiheesta.',
    'Sirkustaiteilija {s} hämmästytti yleisöä upeilla akrobaattinumeroillaan.',
    'Ankara mutta reilu tuomari {s} sai kunnioitusta kaikilta urheilijoilta.',
    'Lentäjä {s} navigoi turvallisesti haastavissa sääolosuhteissa.',
    'Taidegallerian omistaja {s} avasi näyttelyn, joka esittelee nuoria, lahjakkaita taiteilijoita.',
    'Mielenterveysaktivisti {s} piti voimakkaan puheen stigmaa vastaan.',
    '{s} johti kaivauksia, jotka paljastivat muinaisen sivilisaation jäänteitä.',
    '{s} voitti mestaruuden dramaattisen loppukierroksen jälkeen.',
    'Sovelluskehittäjä {s} loi uuden sovelluksen, joka auttaa ihmisiä hallitsemaan aikaansa paremmin.',
    '{s} vei turisteja läpi historiallisen kaupungin, valottaen sen rikasta menneisyyttä.',
    'Matematiikan nero {s} ratkaisi ongelman, joka oli pysynyt ratkaisemattomana vuosikymmeniä.',
    'Lisäksi {s} toimi vapaaehtoisena kotikaupunkinsa ruokapankissa.',
    'Bändin laulaja {s} oli tunnettu ainutlaatuisesta äänestään ja lavakarismastaan.',
    '{s} jakoi kokemuksensa avaruudesta ja sen ihmeistä.',
    '{s} valmisti hienostuneita koruja, joissa yhdistyivät perinteinen käsityö ja moderni muotoilu.',
    '{s} löysi {adj} hyönteislajin ekspeditionsa aikana trooppisessa sademetsässä.',
    'Poliisi {s} ratkaisi monimutkaisen rikostapauksen perusteellisen tutkinnan ansiosta.',
    '{s} voitti viininmaistelukilpailun ja sai tunnustusta poikkeuksellisesta makuaististaan.',
    'Hittisarjan käsikirjoittaja {s} sai kiitosta nokkelista dialogeistaan ja koukuttavasta juonestaan.',
    'Lentokoneinsinööri {s} suunnitteli uuden polttoainetehokkaan lentokoneen mallin.',
    'Kemian opettaja {s} innosti oppilaita interaktiivisilla kokeilla.',
    '{ajd} robotti-insinööri {s} esitteli uuden robotin, joka voi auttaa kotitöissä.',
    'Etiikan professori {s} puhui aiheesta, joka haastoi kuulijat pohtimaan omia moraalisia periaatteitaan.',
    'Valokuvaaja {s} otti kuvia villieläimistä niiden luonnollisessa elinympäristössä.',
    'Luovan kirjoittamisen opettaja {s} auttoi opiskelijoitaan löytämään oman äänensä.',
    'Kokki {s} loi menuun, joka yhdisteli paikallisia raaka-aineita innovatiivisilla tavoilla.',
    'Kuvanveistäjä {s} paljasti uuden patsaan, joka kunnioittaa historiallista henkilöä.',
    'Ilmastonmuutoksen tutkija {s} esitti havaintonsa, jotka herättivät huolta globaalista lämpenemisestä.',
    'Kirjakauppiaan {s} suositusromaanit ovat aina harkittuja ja suosittuja asiakkaiden keskuudessa.',
    'Kemian Nobel-palkinnon voittaja {s} inspiroi nuoria tutkijoita saavutuksillaan.',
    'Nukketeatterin perustaja {s} toi iloa lapsille ainutlaatuisilla esityksillään.',
    'Investointipankkiiri {s} neuvoo suuryrityksiä heidän talousstrategioissaan.',
    'Luokanopettaja {s} juhli opettajansa päivää yhdessä oppilaidensa kanssa.',
    '{s} esiintyi lasten sairaalassa, tuoden iloa potilaiden päiviin.',
    'Motivaatiopuhuja {s} kannusti yleisöä seuraamaan unelmiaan ja tavoittelemaan parastaan.',
    'Graffititaiteilija {s} muutti harmaan sivukujan värikkääksi taideteokseksi.',
    'Kansanterveystyöntekijä {s} ponnisteli lopettaakseen taudin leviämisen yhteisössä.',
    'Data-analyytikko {s} auttoi yrityksiä ymmärtämään suuria tietomääriä paremmin.',
    'Rumpali {s} sai yleisön tanssimaan rytmiensä tahdissa koko yön.',
    'Organisaatiokonsultti {s} auttoi yrityksiä tehostamaan toimintaansa.',
    'Tapahtuman järjestäjänä {s} varmisti, että kaikki sujui suunnitelmien mukaan.',
    'Kuuluisa ruokakriitikko {s} arvosteli ravintolan uuden menun loistavin sanoin.',
    'Vastasyntyneen lapsen vanhemmat, Markku ja Hanna, olivat onnellisia saadessaan syliinsä pienen {s}.',
    'Häätanssinsa harjoitellut {s} esiintyi huikealla tyylillä sulhasensa rinnalla.',
    'Kiertotalouden asiantuntija {s} luennoi siitä, miten meidän tulisi vähentää jätteen tuotantoa.',
    'Alallaan arvostettu lääkäri {s} kehitti uuden hoitomenetelmän syöpäsairauksien torjumiseksi.',
    'Ansiokkaasta työstään maahanmuuttajien parissa {s} sai vuoden kansalaisvaikuttaja -palkinnon.',
    'Muusikko {s} julkaisi odotetun uuden albuminsa ja fanit ympäri maailmaa riemuitsivat.',
    'Voimistelija {s} saavutti huippupisteet tasapainohipassa ja voitti kultamitalin.',
    'Muotisuunnittelija {s} esitteli näytöksessään luomuksia, jotka ihastuttivat yleisöä.',
    'Mainosalalla menestynyt luova suunnittelija {s} sai idean uuteen kampanjaan lenkillä ollessaan.',
    'Leipuri {s} loihti {adv} herkullisen kakun, joka hämmensi kaikkia huikealla ulkonäöllään.',
    'Kaikkien suosima näyttelijä {s} esiintyi hittimusikaalissa taitavasti ja saavutti valtavaa suosiota.',
    'Jääkiekkoilija {s} teki tärkeän maalin ratkaisevassa finaalissa ja auttoi joukkueensa mestaruuteen.',
    'Luonnontieteilijä {s} löysi uuden uhanalaisen lajin, jonka olemassaolosta ei oltu aiemmin tiedetty.',
    'Yritysjohtaja {s} ilmoitti suunnitelmastaan lanseerata uusi tuote markkinoille.',
    'Kokki {s} voitti kilpailun upeilla makuyhdistelmillään ja kruunattiin voittajaksi.',
    'Projektipäällikkö {s} järjesti onnistuneen yrityksen tiimipäivän, jossa oli hauskaa ja opittiin uutta.',
    'Politiikko {s} puolusti kiivaasti tärkeää asiaa ja sai kansalaiset ajattelemaan uudelleen näkökulmiaan.',
    'Journalisti {s} kirjoitti rohkean artikkelin, joka paljasti suuren skandaalin.',
    'Teollinen muotoilija {s} suunnitteli toimivan ja tyylikkään uutuustuotteen, joka keräsi paljon kehuja.',
    'Autonkuljettaja {s} vei asiakkaansa turvallisesti ja mukavasti perille.',
    'Maalari {s} loi upeita taideteoksia, jotka herättivät tunteita katsojissa.',
    'Kirjailija {s} julkaisi pitkään odotetun uuden romaaninsa, joka nousi heti bestseller-listalle.',
    'Ekonomi {s} loi liiketoimintasuunnitelman, joka herätti sijoittajien mielenkiinnon.',
    'Tapahtumajärjestäjä {s} loi ikimuistoiset puitteet unohtumattomalle juhlalle.',
    'Opiskelija {s} suoritti tentin loistavin tuloksin ja sai kiitosta valmiuksistaan.',
    'Viherrakentaja {s} suunnitteli upean puutarhan, joka valittiin vuoden kauneimmaksi.',
    'Valokuvaaja {s} ikuisti kauniin auringonlaskun upealla otoksellaan.',
    'Tuotantopäällikkö {s} varmisti, että tuotteet valmistuivat ajoissa ja laadukkaasti.',
    'Myyntineuvotteluiden jälkeen {s} saavutti yhteistyösopimuksen arvostetun asiakkaan kanssa.',
    'Urheilulääkäri {s} auttoi kilpailijaa palaamaan nopeasti takaisin huippukuntoon.',
    'Asiantuntija {s} jakoi arvokkaita vinkkejä alansa opiskelijoille {adv}.',
    '{ajd} palomies {s} pelasti kissan puusta ja sai kiitosta rohkeasta toiminnastaan.',
    'Kirjanpitäjä {s} hoiti yrityksen taloushallinnon ammattitaitoisesti ja huolellisesti.',
    'Arkkitehti {s} suunnitteli kaupunkiin upean pilvenpiirtäjän, joka kohotti kaupungin profiilia.',
    'Fysioterapeutti {s} auttoi potilasta kuntoutumaan loukkaantumisen jälkeen.',
    'Tietokoneohjelmoija {s} kehitti innovatiivisen sovelluksen, joka helpotti arkipäivän rutiineja.',
    'Verkkokauppiaana {s} menestyi hyvin ja sai suuren asiakaskunnan.',
    'Kielenkääntäjä {s} välitti tärkeän viestin sujuvasti eri kielillä.',
    'Huippukokki {s} loihti fine dining -menuun uusia jännittäviä makuyhdistelmiä.',
    'Rehtori {s} järjesti hienosti koulun päättäjäisjuhlan ja onnitteli valmistuvia opiskelijoita.',
    'Kahvilayrittäjä {s} tarjoili herkullista kahvia ja herkkuja asiakkaille.',
    'Teatteriohjaaja {s} loi näytelmästä hienostuneen kokonaisuuden, joka sai kiitosta kriitikoilta.',
    'Historiantutkija {s} kirjoitti mielenkiintoisen artikkelin menneiden aikojen tapahtumista.',
    'Muusikkopariskunta {s} teki kauniin kappaleen yhdessä, joka valloitti radiokanavat.',
    'Kauppatieteiden professori {s} piti luentoa aiheesta, joka herätti kiinnostusta opiskelijoiden keskuudessa.',
    'Toimittaja {s} keräsi arvokasta tietoa ja haastatteli mielenkiintoisia henkilöitä uutisjuttua varten.',
    'Sosiaalityöntekijä {s} auttoi asiakasta vaikean elämäntilanteen keskellä ja tarjosi tukea.',
    'Uutisissa kerrottiin, kuinka valokuvaaja {s} voitti arvostetun valokuvakilpailun upeilla otoksillaan.',
    'Kestävään kehitykseen tähtäävän uuden asumisratkaisun suunnitteli arkkitehti {s}, joka palkittiin innovatiivisesta työstään.',
    'Harvinaista sairautta sairastavat potilaat saivat uutta toivoa, kun huippulääkäri {s} kehitti heille uudenlaisen hoitomenetelmän.',
    'Yrittäjäksi ryhtynyt nuori {s} onnistui lanseeraamaan menestyksekkään startup-yrityksen, joka keräsi merkittävän rahoituksen.',
    'Kirjailija {s} julkaisi odotetun uutuusromaaninsa, joka nousi välittömästi bestseller-listan kärkeen.',
    'Urheilija {s} ylitti itsensä ja voitti kultamitalin kansainvälisessä kilpailussa.',
    'Haastattelussa kuuluisa näyttelijä {s} paljasti tulevasta roolistaan blockbuster-elokuvassa.',
    'Maailman johtajat kohtaavat tärkeässä neuvottelussa, kun kansainvälinen diplomatia kukoistaa ja {s} edustaa maataan.',
    'Suositun artistin {s} stadionkonsertti toteutetaan fanien toiveiden pohjalta.',
    'Suosittu {ajd} tv-sarja palaa ruutuun uuden kauden myötä, ja {s} näyttelee jälleen pääroolissa.',
    'Uutisissa kerrottiin, että naapurin {s} sai upeita valokuviaan myymälänäyttelyyn.',
    'Veljeni suunnitteli uuden kodin, joka on ympäristöystävällinen ja moderni, kuten {s} halusi.',
    'Eräänä päivänä koirapuistossa tapasin ihanan henkilön, {s}, joka keksi uudenlaisen leikkivälineen koirille.',
    'Paras ystäväni {s} keksi nerokkaan tavan tehdä terveellistä ja herkullista aamupalaa.',
    'Viime viikolla tuttavani {s} voitti lotossa miljoonan euron päävoiton.',
    'Siskoni {s} julkaisi odotetun romaaninsa, joka saavutti suuren suosion kirjallisuuspiireissä.',
    'Tuttuni {s} ylitti kaikki odotukset ja voitti viimeisen kappaleen tarjouskilpailussa.',
    'Äitini {s} osallistui kansainväliseen neuvottelutilaisuuteen, jossa tuli tunnustetuksi asiantuntijana omalta alaltaan.',
    'Paras ystäväni {s} sai pitkään odotetun keikkatarjouksen suosikkibändiltään.',
    'Veljeni {s} jatkaa tv-sarjan suurta suosiota uuden kauden myötä, ja hänen roolinsa on yhä keskeisempi tarinassa.',
    'Uutisissa kerrottiin, että naapurin {s} sai upeita valokuviaan myymälänäyttelyyn, mikä sai paikalliset taiteenharrastajat innostumaan.',
    'Veljeni suunnitteli uuden kodin, joka on ympäristöystävällinen ja moderni, ja {s} oli tyytyväinen lopputulokseen.',
    'Eräänä päivänä koirapuistossa tapasin ihanan henkilön, joka keksi uudenlaisen leikkivälineen koirille, ja {s} halusi jakaa idean kaikkien kanssa.',
    'Paras ystäväni keksi nerokkaan tavan tehdä terveellistä ja herkullista aamupalaa, mistä {s} innostui kokeilemaan sitä omassa arjessaan.',
    'Viime viikolla tuttavani voitti lotossa miljoonan euron päävoiton, minkä kuullessaan {s} oli täysin häkeltynyt.',
    'Siskoni julkaisi odotetun romaaninsa, joka saavutti suuren suosion kirjallisuuspiireissä, ja {s} oli ylpeä siskonsa saavutuksesta.',
    'Tuttuni ylitti kaikki odotukset ja voitti viimeisen kappaleen tarjouskilpailussa, mikä yllätti myös {s} {adv}.',
    'Äitini osallistui kansainväliseen neuvottelutilaisuuteen, jossa tuli tunnustetuksi asiantuntijana omalta alaltaan, minkä ansiosta myös {s} sai uusia liikekumppanuuksia.',
    'Paras ystäväni sai pitkään odotetun keikkatarjouksen suosikkibändiltään, mikä sai {s} juhlatuulelle.',
    'Veljeni jatkaa tv-sarjan suurta suosiota uuden kauden myötä, ja hänen roolinsa on yhä keskeisempi tarinassa, mikä ilahdutti myös {s} suuresti.',
    'Kun saavuin paikalle, huomasin, että rakennustyömaalla oli suuri sotku, ja {s} ryhtyi heti järjestämään töitä kuntoon.',
    'Projektinjohtajan tehtäväksi annettiin korjata rakennustyömaalla vallitseva suuri sotku, ja {s} tarttui innokkaasti haasteeseen.',
    'Rakennustyömaalla vallitseva suuri sotku sai työntekijät hämmennyksen valtaan, mutta {s} löysi ratkaisun tilanteeseen.',
    'Vastuullisen työmaapäällikön tehtävänä oli hoitaa suuri sotku rakennustyömaalla kuntoon, ja {s} ryhtyi ripeästi toimiin.',
    'Suuresta sotkusta huolestuneet työntekijät kääntyivät rakennustyömaan työnjohtajan puoleen, ja {s} lupasi hoitaa tilanteen kuntoon.',
    'Rakennustyömaalla oli paha sotku, josta {s} vastasi ja pyrki välittömästi korjaamaan tilanteen.',
    'Sotkuinen tilanne rakennustyömaalla herätti pahennusta, ja {s} otti ohjat käsiinsä, jotta saataisiin aikaan muutos parempaan suuntaan.',
    'Rakennustyömaan suuri sotku aiheutti viivästyksiä, mutta {s} pani kaikkensa likoon, jotta aikataulut saataisiin kiinni.',
    'Joukkueenjohtaja koki suurta pettymystä, kun rakennustyömaa osoittautui sotkuiseksi, mutta {s} ei antanut periksi vaan alkoi välittömästi siivota.',
    'Suuri sotku rakennustyömaalla osoitti puutteellista työjärjestystä, johon {s} puuttui tehokkaasti ja asetti uudet toimintasuunnitelmat.',
    'Kun saavuin puistoon, huomasin, että siellä metelöitiin kovasti, ja {s} päätti puuttua tilanteeseen.',
    'Puiston rauhaa häiritsi metelöinti, ja {s} otti tilanteen hoitaakseen ja pyrki luomaan rauhallisen ilmapiirin.',
    'Metelöinti puistossa aiheutti närää ympäristössä oleville ihmisille, ja {s} päätti selvittää, mistä meteli johtui ja pyrki rauhoittamaan tilanteen.',
    'Kun paikalle saapui valitus puistossa metelöinnistä, {s} ryhtyi toimiin ja etsi tapoja vähentää melua ja varmistaa, että puistossa vallitsi rauhallinen tunnelma.',
    'Paikalliset asukkaat olivat turhautuneita puistossa tapahtuvaan metelöintiin, ja {s} teki aloitteen järjestää tilaisuus, jossa asiasta voitaisiin keskustella ja löytää yhteisiä ratkaisuja.',
    'Metelöinti puistossa aiheutti huolta alueen asukkaille, ja {s} otti tilanteen vakavasti ja teki valituksen kaupungille asiasta.',
    'Metelöinti puistossa sai monet puistonkävijät ärsyyntymään, mutta {s} ryhtyi toimiin ja teki aloitteen asukkaiden rauhanomaisesta yhteistyöstä metelin vähentämiseksi.',
    'Puistossa vallitseva metelöinti herätti pahennusta lähialueen asukkaissa, ja {s} lähestyi paikallisia viranomaisia tilanteen rauhoittamiseksi.',
    'Yleinen mielipide metelöinnistä puistossa nousi esille, ja {s} päätti ottaa asiakseen tehdä kampanjan puiston rauhan säilyttämiseksi.',
    'Puistossa jatkuva metelöinti vaikutti negatiivisesti sen rauhalliseen tunnelmaan, ja {s} pyrki luomaan aloitteen puistokävijöiden tietoisuuden lisäämiseksi hiljaisuuden arvostuksesta.',
    'Kalakauppiaalla oli {adj} päivä, sillä asiakkaita oli paljon, ja {s} oli iloinen saadessaan palvella heitä {adv}.',
    'Kalakauppiaan liike vilisi asiakkaita hyvän päivän ansiosta, ja {s} piti huolen siitä, että kaikki saivat parasta palvelua.',
    'Kalakauppiaan myymälä oli täynnä asiakkaita, sillä päivä oli vilkas, ja {s} tyytyväisenä huomasi suosion.',
    'Kauppiaan myyntitiskille oli jonoa, sillä asiakkaita oli paljon, ja {s} iloitsi suosiostaan.',
    'Kalakauppias ei voinut olla tyytyväisempi, sillä asiakkaita virtasi liikkeeseen ja {s} tarjoili tuoretta kalaa innolla.',
    'Liikkeeseen tulvi asiakkaita, jotka olivat kiinnostuneita kalatuotteista, ja kalakauppias {s} oli tyytyväinen suureen asiakasvirtaan.',
    'Kalakauppias teki erinomaista myyntiä, sillä asiakkaita kävi paljon, ja {s} oli tyytyväinen työnsä tuloksiin.',
    'Päivä oli kiireinen kalakauppiaalle, mutta samalla myös onnistunut, kun asiakkaita kävi paljon, ja {s} oli tyytyväinen liikkeensä suosioon.',
    'Asiakasmäärä kalakaupassa oli huomattava, ja kalakauppias {s} hoiti myyntiä ripeästi ja tehokkaasti.',
    'Kalakauppiaan myymälässä vallitsi vilkas tunnelma, kun asiakkaita oli paljon, ja {s} huolehti kaikkien tarpeista.',
    'Kaupungin aurauskalusto oli vioittunut, ja {s} ryhtyi heti korjaamaan sitä, jotta lumiset tiet saataisiin aurattua.',
    'Lumisessa kaupungissa oli ongelma, kun aurauskalusto oli vioittunut, mutta {s} oli päättänyt selvittää tilanteen ja saada kadut siistiksi.',
    'Talvimyrsky yllätti, mutta ikävä kyllä kaupungin aurauskalusto oli vioittunut, mikä aiheutti haasteita liikenteelle. {s} pani kaikkensa likoon, jotta tilanne saataisiin korjattua nopeasti.',
    'Kaupungin aurauskalustossa ilmeni vika, ja {s} järjesti ripeän huoltotiimin paikalle korjaamaan tilanteen.',
    'Kun lunta satoi tiuhaan tahtiin, kaupungin aurauskalusto joutui odottamattomaan vikaan, mutta onneksi {s} oli valmis auttamaan ja huolehtimaan siitä, että liikenne saatiin jälleen sujumaan.',
    'Lumimyräkkä teki tuloaan, mutta valitettavasti kaupungin aurauskalusto oli vioittunut. Tilanteeseen puuttui nopeasti {s}, joka etsi ratkaisuja lumisten teiden avaamiseksi.',
    'Vioittunut aurauskalusto aiheutti {adv} haasteita kaupungin tienhoidolle, mutta {s} oli valmiina ryhtymään toimiin ja organisoi tarvittavat korjaukset.',
    'Kaupungin aurauskalustossa ilmeni vika juuri lumipyryn aikana, mikä hankaloitti liikkumista. {s} otti asiakseen järjestää korjaustöitä, jotta liikenne saataisiin jälleen sujuvaksi.',
    'Aurauskaluston vioittuminen tuli pahaan aikaan lumimyrskyn aikana, mutta {s} ei antanut periksi vaan pyrki nopeasti saamaan kaluston toimintakuntoon.',
    'Talvimyrsky asetti kaupungin aurauskalustolle haasteita, kun se vioittui, mutta {s} oli valmis käyttämään kaikki mahdolliset resurssit, jotta katujen aurauksesta huolehdittaisiin.',
    'Koulujen väliset hiihtokilpailut järjestettiin hiihtomaassa, ja {s} toimi kilpailujen pääjärjestäjänä.',
    'Koulujen väliset hiihtokilpailut päätettiin järjestää hiihtomaassa, ja {s} oli innokas osallistumaan kilpailujen suunnitteluun ja toteutukseen.',
    'Hiihtomaan upeat ladut tarjosivat {adv} täydellisen taustan koulujen välisille hiihtokilpailuille, jotka järjestettiin onnistuneesti {s} toimiessa kilpailujen koordinaattorina.',
    'Kouluista peräisin olevat hiihtolupaavat osallistuivat innolla hiihtokilpailuihin, jotka järjestettiin upeassa hiihtomaassa {s} toimiessa tapahtuman järjestäjänä.',
    'Hiihtomaassa järjestetyissä koulujen välisissä hiihtokilpailuissa {s} vastasi kilpailujen sujuvuudesta ja tunnelman luomisesta.',
    'Kaikki osallistujakoulut innostuivat koulujen välisistä hiihtokilpailuista, jotka järjestettiin hiihtomaassa, ja {s} auttoi kilpailujen valmistelussa.',
    'Hiihtomaan upeat maisemat ja laskettelukeskus toivat täydelliset puitteet koulujen välisille hiihtokilpailuille, jotka sujuivat erinomaisesti {s} johdolla.',
    'Koulujen väliset hiihtokilpailut olivat odotettu tapahtuma, joka järjestettiin upeassa hiihtomaassa ja sai aikaan hiihtourheilun huuman. {s} vastasi kilpailujen kokonaisjärjestelyistä.',
    'Koululaiset odottivat innolla koulujen välisten hiihtokilpailujen järjestämistä hiihtomaassa, ja {s} toteutti tapahtuman sujuvasti.',
    'Hiihtomaan loistavat hiihtoladut tarjosivat täydellisen areenan koulujen välisille hiihtokilpailuille, joissa {s} vastasi kilpailujen onnistumisesta.',
    'Liikenne keskeytyi tietyön takia, mikä sai asukkaat harmistumaan. {s} sanoi: "En ymmärrä miksi katuja pitää korjata keskellä kesää."',
    'Kadun remontti aiheutti {adv} liikenteen katkeamisen ja asukkaat olivat ärsyyntyneitä. {s} kommentoi: "Miksi ihmeessä katuja aletaan korjata kesäaikaan?"',
    'Liikenne oli tukossa, kun tietyö tehtiin, ja se sai asukkaat turhautumaan. {s} huudahti: "Mikä idea on korjata katuja keskellä kesää?!"',
    'Kadun korjaustyöt keskeyttivät liikenteen ja asukkaat olivat pettynyttä mieltä. {s} kysyi ääneen: "Miksi juuri nyt pitää ruveta katuja korjaamaan? Kesähän on vilkkainta aikaa liikenteessä!"',
    'Tietyö haittasi liikennettä ja asukkaat valittivat. {s} sanoi: "Onko täysin välttämätöntä korjata katuja kesäaikaan?"',
    'Kadun korjaukset aiheuttivat liikenteen seisahduksen ja asukkaat olivat ärtyneitä. Asukas {s} ihmetteli ääneen: "Miksi katujen korjaamista ei voisi tehdä muulloin kuin kesällä?"',
    'Liikennettä haittasi tietyö ja asukkaat olivat tyytymättömiä. Heidän keskuudessaan kuului kommentti: "Eikö katuja voisi korjata kesälle epätyypillisenä aikana? Se estäisi liikenteen ruuhkautumista." totesi {s}.',
    'Asukkaat olivat harmistuneita, kun liikenne katkesi tietyön takia keskellä kesää. Paikallinen asukas {s} sanoi pettymyksestä äänensävyssään: "Eikö tietyöt voisi suunnitella niin, ettei ne vaikuttaisi kesäkauden liikenteeseen?"',
    'Liikenteen katkaisu tietyön vuoksi ärsytti asukkaita. Pitkäaikainen asukas {s} huomautti: "Katuja olisi voitu korjata aikaisemmin tai myöhemmin, eikä keskellä vilkkainta kautta!"',
    'Tietyökeskeytys aiheutti liikennettä, ja asukkaat olivat turhautuneita. Kaupungin insinööri {s} kommentoi asiaa että "Kadun korjaus on välttämätöntä, mutta ymmärrän, että se aiheuttaa haittaa."',
    'Haluan antaa kiitosta erinomaisesta palvelustanne. Asiakaspalvelijanne {s} oli erittäin avulias ja ystävällinen. Sain tarvitsemani tiedon nopeasti ja vaivattomasti. Iso kiitos hyvästä kokemuksesta!',
    'Olen erittäin tyytyväinen ostokseeni ja haluan kiittää myyjää {s}. Hän oli asiantunteva, kärsivällinen ja avulias. Hän auttoi minua löytämään juuri oikean tuotteen tarpeisiini. Palvelunne oli erinomaista, kiitos!',
    'Kiitokset ravintolan henkilökunnalle upeasta illasta! Ruoka oli herkullista ja tarjoilija {s} teki kaikkensa varmistaakseen, että meillä oli hyvä kokemus. Hän oli ammattitaitoinen ja huomaavainen, ja asiakaspalvelunne sai meidät tuntemaan itsemme tervetulleiksi. Suuret kiitokset!',
    'Haluan kiittää {s} erinomaisesta työstä auton korjaamisessa. Hän oli luotettava ja tarkka työssään. Autoni palautettiin kuntoon sovittuna aikana, ja huoltoprosessi oli sujuva. Kiitos ammattitaitoisesta palvelusta!',
    'Olen todella kiitollinen saamastani tuesta. {s} oli erittäin ystävällinen ja auttoi minua ratkaisemaan ongelmani. Hän kuunteli kärsivällisesti ja tarjosi hyödyllisiä neuvoja. Iso kiitos avusta, arvostan sitä todella!',
    'Kiitos todella hyvästä hoitokokemuksesta. Hoitaja {s} oli empaattinen ja huolehtiva. Hänellä oli loistava ammattitaito ja hän teki kaikkensa varmistaakseen, että minulla oli mukava olo. Iso kiitos hyvästä hoidosta!',
    'Haluan kiittää palveluasentajaanne {s} erinomaisesta työstä kotini korjauksessa. Hän oli taitava ja nopea työssään. Huolellisuutensa ansiosta ongelma ratkesi, ja lopputulos oli erinomainen. Kiitos ammattitaitoisesta palvelusta!',
    'Iso kiitos mahtavasta tapahtumasta! Tapahtumajärjestäjänne {s} teki huikeaa työtä sujuvoittaessaan kaikkea. Hän oli ammattimainen ja järjestelmällinen, ja tapahtuma sujui moitteettomasti. Kiitos upeasta kokemuksesta!',
    'Kiitos loistavasta opetuksesta! Opettajanne {s} oli inspiroiva ja taitava. Hän innosti ja kannusti oppilaita sekä jakoi tärkeää tietoa selkeällä tavalla. Olen erittäin tyytyväinen opetukseenne, kiitos paljon!',
    'Kiitos fantastisesta hotellikokemuksesta! Henkilökuntanne, erityisesti vastaanottovirkailija {s}, teki vierailustani ikimuistoisen. Hän oli ystävällinen, avulias ja tehokas. Palvelunne oli ensiluokkaista, iso kiitos siitä!',
    '{s} muutti harmaan talon värikkääksi taideteokseksi.',
    'Suuri esikuvamme {s} ponnisteli lopettaakseen taudin leviämisen afrikassa.',
    'Meille vielä tuntematon nero {s} auttoi opiskelijoita ymmärtämään suuria tietomääriä paremmin.',
    'Kitaristi {s} sai yleisön tanssimaan samassa tahdissa koko aamun.',
    'Astetta vahvempi {s} auttoi nostokurkea kohottamaan järkäleen.',
    'Sirkuksen järjestäjänä {s} varmisti, ettei kaikki sujuisi suunnitelmien mukaan.',
    'Kuuluisa muusikko {s} arvosteli ravintolan uuden menun kärkevin sanoin.',
    'Koululaisen vanhemmat olivat onnellisia saadessaan syliinsä pienen {s} tekemän possu-taulun.',
    'Soittamaan {adv} harjoitellut {s} esiintyi tyhjälle salille.',
    'Kierrätysekspertti {s} luennoi siitä, miten pahvia voi uudelleenkäyttää kotioloissa.',
    'Tämä ei välttämättä toimi, totesi {s} kehitettyään uuden mullistavan hoitomenetelmän syöpäsairauksien torjumiseksi.',
    '{s} sai vuoden kansalaiskansalainen -palkinnon. Se on {ajd} tunnustus.',
    'Tuottaja {s} muistaa {adv} kuinka fanit ympäri maailmaa riemuitsivat orkesterin menestyksen huipulla.',
    'Vaikea on ymmärtää miksi {s} ei voinut vastata kysymykseeni vaikka huusin kovaa.',
    'Asentaja {s} purki laittoman sähköasennuksen.',
    'Mainosalalla kaiken kokoenut entinen suunnittelija {s} sai lahjaksi tomaatin.',
    'Naapurimme {s} leipoi {adv} herkullisen pullan.',
    'Kaikkien suosima tanssija {s} esiintyi jääshowssa ensi kertaa.',
    'Runoilija {s} teki tärkeän huomion liittyen nakertajiin.',
    'Tutkija {s} löysi uuden uhanalaisen lajin, joka asui vintillä.',
    'Keksijä {s} lanseeri uuden tuotteensa.',
    'Juoksija {s} voitti kilpailun upeilla jaloillaan.',
    'Konsultti {s} järjesti onnistuneen yrityksen tiimipäivän, jossa oli tylsää ja opittiin vähän.',
    'Johtaja {s} kritisoi uutta lakia.'
]

SENTENCES_STREETS = [
    'Toivotaan energiahuoltoa {s} 6 sähköt roikkuu sähkölinjalla.',
    'Osoitteessa {s} tien puolella on kivi.',
    'Kotikadullamme on suuri lumikasa osoitteessa {s} 100, 00100 Helsinki.',
    'Terveisin Erkki Esimerkki, {s} 16 A 199, Helsinki.',
    'Terveisin Jaska, {s}, Helsinki.',
    'Vastaanottaja: Nimi Henkilön, {s} 10 Helsinki.',
    '{s} risteyksessä on puu vinossa.',
    'Keskustassa {s} kulmassa on suuri lumikasa kadulla.',
    'Rannan tuntumassa {s} 9 on lokki, joka on suuri kuin pieni talo tai suuri muurahainen.',
    'Osoitteeni on {s} 1, 00100 Helsinki.',
    'Voisiko {s} saada korjauksen aitaan.',
    'Kadullamme osoitteessa {s} tarvitaan aurausta.',
    '{s} on huonossa kunnossa talvella.',
    'Työmaa tukkii liikennettä {s}.',
    'Auraus ei tullut tiistaina {s} 12.',
    '{s} 1 aurataan liian usein.',
    'Malmin {s} 5 A 1 varpuset nokkii murusia.',
    'Toivon aurausta {s}lle.',
    'AsOy {s}.',
    '{s} risteykseen kerääntyy sateella suuri lammikko vettä.',
    'Karhujen kerhotalo {s} 11 B ottaa vastaan reippaita urheilijoita ympäri vuoden.',
    'On valitettavaa, että {s} 8:n jalkakäytävä on niin epätasainen.',
    'Ymmärrän, että {s} 15:n remontti on tarpeen, mutta melu on sietämätöntä.',
    'Iltaisin {s} 22:n katuvalot eivät toimi, mikä tekee kulkemisesta turvatonta.',
    'Eikö {s} 30:n puiston penkkejä voitaisi puhdistaa useammin?',
    'Olisi tärkeää, että {s} 5:n leikkipaikan hiekkalaatikko vaihdettaisiin turvallisempaan materiaaliin.',
    'Talvisin {s} 12:n loppupään katuvalaistus on aivan riittämätön.',
    'Toivon, että {s} 17:n bussipysäkille saataisiin katoksen lisäksi penkit odottaville.',
    'Pysäköinti {s} 3:n varrella on aina yhtä haasteellista, tarvitsemme lisää tilaa.',
    'Löytyisikö ratkaisu {s} 24:n nopeasti täyttyvien roskisten ongelmaan?',
    'Olen huolissani siitä, että {s} 2:n yli kulkeva silta näyttää rapistuneelta.',
    'Kiitos, että {s} 14:n leikkipuistoa on kunnostettu – lapset rakastavat sitä!',
    'Milloin {s} 9:n katukuoppia aiotaan paikata? Ne ovat vaaraksi pyöräilijöille.',
    'Olisi hienoa, jos {s} 19:n koirapuistoon voitaisiin asentaa lisää varjoa tarjoavia puita.',
    'Haluaisin antaa palautetta {s} 7:n varren loistavasta katutaidegalleriasta.',
    'Voisiko {s} 27:n risteysalueelle saada lisävalaistusta?',
    'Pysäköintikiellot {s} 13:ssa ovat epäselviä ja aiheuttavat sekaannusta.',
    'Olen iloinen siitä, että {s} 29:n katualue on saanut uutta asfalttia.',
    'Voisivatko liikennevalot {s} 16:ssa vaihtua sujuvammin, ettei ruuhkia syntyisi niin paljon?',
    'On pettymys, että {s} 21:n puiston kukkapenkit on annettu rikkaruohottua.',
    'Kiitokset hyvin hoidetusta {s} 34:n bussipysäkistä, se on aina siisti.',
    'Kehotan kaupunkia tarkistamaan {s} 10:n viemäröinnin, joka tuoksuu epämiellyttävälle.',
    'Pyydän, että {s} 26:een saataisiin parempia pyörätelineitä.',
    'Kiitän siitä, että {s} 20:n kadunvarsipuut on leikattu ja ne näyttävät upeilta.',
    'Miksei {s} 18:n varrella olevia roskiksia tyhjennetä tarpeeksi usein?',
    '{s} 35:n leikkialueen turvallisuus on todella parantunut viime aikoina, kiitos siitä.',
    'Toivoisin, että {s} 31:ssä järjestettäisiin useammin katutapahtumia.',
    'On harmillista, että {s} 23:n ajojärjestelyt ovat niin sekavat.',
    'Hyvä, että {s} 4:n leikkipuistoon on lisätty uusia laitteita.',
    'Onko {s} 28:n kulkutien ylläpidosta vastaava taho tiedossa? Se kaipaa kipeästi kunnostusta.',
    'Olisi tärkeää saada lisää polkupyöräkaistoja {s} 42:lle.',
    'Miksi {s} 36:n suojateiden maalaukset ovat niin haalistuneet?',
    'Onko {s} 25:llä oleva graffitiseinä virallinen? Se näyttää upealta!',
    'Pyydän, että {s} 41:n lasten turvallisuus koulumatkalla taataan paremmin.',
    'Ilahduin nähdessäni, että {s} 40:n alueella on otettu käyttöön uusia biojäteastioita.',
    'Tunnen huolta {s} 6:n talojen julkisivujen rapistumisesta, onko korjaussuunnitelmaa?',
    'Olisi suotavaa, että {s} 32:n risteyksessä oleva pyörätie levennettäisiin.',
    'Haluaisin kysyä, onko {s} 38:n katusoitto sallittua vai ei?',
    'Ilokseni {s} 1:n alueella on nyt enemmän viheristutuksia ja se näyttää viihtyisämmältä.',
    'On valitettavaa, että {s} 43:n koulun vieressä oleva rakennustyömaa aiheuttaa niin paljon pölyä.',
    'Toivoisin, että {s} 33:n tietyöt saataisiin päätökseen nopeammin liikenteen sujuvuuden vuoksi.',
    'Olen erittäin tyytyväinen {s} 37:n jalkakäytävien talvikunnossapitoon, kiitos hyvästä työstä.',
    'Olisiko {s} 44:ään mahdollista saada lisää katuvalaisimia, varsinkin talvikuukausina?',
    'Kiitos, että {s} 30:n alueen puistot ovat pidetty siistinä ja turvallisina.',
]

SENTENCES_AREAS = [
    '{s} alueen kehittäminen pitäisi olla etusijalla.',
    '{s} nauttii taas kesäisestä auringosta.',
    '{s} on kaupungin paras alue.',
    '{s} siltakadulla liikennettä estää lumiukko.',
    'T: Kalle Ankka, {s} Helsinki.',
    'Yst. Aku Ankka, {s}.',
    'Voisiko {s} saada puiston.',
    '{s} rakennukset on suojeltava. ',
    'Kova meteli vaivaa {s}, voisko alueen siirtää?',
    'Koulu tarvitaan {s}.',
    'Tieremontti {s} piinaa autoilijoita.',
    '{s} kirjasto on tosi hyvä!',
    '{s} käsityökerho iloitsee uusista materiaaleista.',
    'Jo viime vuonna {s} aurauksessa oli parannettavaa.',
    '{s} kerhotalon pysäköinti ei yksinkertaisesti toimi.',
    'Huomasin, että {s} leikkipuiston hiekkalaatikko kaipaa täyttämistä.',
    'Valitettavasti {s} koirapuiston aidassa on aukko, josta koirat pääsevät karkaamaan.',
    'Täytyy kehua, että {s} kirjasto on aina niin siisti ja hyvin järjestetty.',
    'Olen pettynyt, ettei {s} uimahallin aukioloaikoja ole laajennettu.',
    'Olisi hienoa, jos {s} puistoihin saataisiin lisää roskiksia.',
    'Kiva nähdä, että {s} katutaideprojektit tuovat iloa alueelle.',
    'Haluaisin ehdottaa, että {s} asukaspysäköintiä voisi parantaa uusilla merkinnöillä.',
    'Huomasin, että {s} alikulkutunnelin valaistus on pahasti rikki.',
    'Pitäisikö {s} jalkakäytävät saada parempaan kuntoon? Ne ovat paikoitellen vaarallisia.',
    'Kiitos, että {s} lasten leikkipaikat on pidetty niin turvallisina ja puhtaina.',
    'Toivottavasti {s} koulureitin turvallisuuteen kiinnitetään pian huomiota.',
    'Ihmettelen, miksi {s} bussipysäkkien katokset eivät pidä sadetta.',
    'Olisi upeaa, jos {s} alueelle saataisiin enemmän pyöräteitä.',
    'Olen huolissani siitä, että {s} valaistus on liian himmeää iltaisin.',
    'Voisiko {s} vanhoja rakennuksia suojella paremmin?',
    '{s} kauppakeskuksen ilmainen wifi on erittäin hyvä palvelu!',
    'Näyttää siltä, että {s} jätehuolto voisi toimia tehokkaammin.',
    'On todella hienoa, että {s} asukkaat ovat niin ystävällisiä ja avuliaita.',
    'Olen huomannut, että {s} alueen nopeusrajoituksia ei valitettavasti noudateta.',
    'Olisi mukavaa, jos {s} puistoissa järjestettäisiin enemmän tapahtumia.',
    'Toivon, että {s} katuvalojen huolto saataisiin ajantasalle.',
    'Kehun mielelläni {s} kaupunginosan järjestämää kesäkonserttisarjaa.',
    'Olisi hyvä, jos {s} alueen tyhjiä liiketiloja voisi hyödyntää paremmin.',
    'Harmillista, että {s} julkiset vessat ovat usein likaisia tai rikki.',
    'Toivoisin, että {s} puistokaduille tulisi enemmän penkkejä istuskelua varten.',
    'Ymmärrän, että {s} rakennustyömaasta aiheutuu melua, mutta voitaisiinko sitä rajoittaa iltaisin?',
    'Mielestäni {s} juna-aseman ympäristö voisi olla turvallisempi.',
    'Onko mahdollista lisätä liikennemerkkejä {s} risteyksiin parantamaan näkyvyyttä?',
    'Haluaisin kiittää {s} alueen katujen kunnossapidosta talven aikana.',
    'Onko tietoa, milloin {s} vesijohtoverkoston vuotava kohta korjataan?',
    'Olisi hienoa, jos {s} asukkaille tarjottaisiin yhteisökompostointimahdollisuus.',
    'En ole tyytyväinen siihen, miten {s} parkkialueen liikennejärjestelyt on suunniteltu.',
    'Toivon, että {s} alueen katutöistä tiedotetaan paremmin etukäteen.',
    '{s} naapurusto on kaunis, mutta kaduilla lojuu valitettavan paljon roskaa.',
    'Kiitos, että {s} kunnossapitohenkilöstö poistaa nopeasti graffitit.',
    '{s} alueen lehtien ja oksien keräyspisteitä pitäisi olla enemmän.',
    'Kaipaisin {s} torille lisää viikoittaisia torimyyjiä.',
    'Olisi arvokasta, jos {s} historiallista miljöötä hyödynnettäisiin enemmän.',
    'Minua ärsyttää, että {s} alueella on niin paljon luvatonta mainosjakelua.',
    'On mahtavaa, kuinka {s} yhteisöllisyys näkyy asukkaiden kesken.',
    'Voisitteko kiinnittää huomiota {s} liikennevalojen toimivuuteen?',
    'Todella kiva, että {s} alueelle on saatu uusia puunkatveksessa olevia kahviloita.',
    'Toivoisin, että {s} alueen leikkipaikkojen liukumäet tarkistettaisiin turvallisuuden vuoksi.',
    'Harmi, että {s} vesialueen ympärillä on niin vähän istumapaikkoja.',
    'Pitäisikö {s} viheralueiden hoitoa parantaa? Monet pensaat näyttävät hoitamattomilta.',
    'Onko suunnitteilla, että {s} alueen kouluihin tulisi lisää iltapäiväkerhotoimintaa?',
    'Olisi kiva, jos {s} asukkaille järjestettäisiin enemmän yhteisiä siivoustalkoita.'
]

EVALUATION_SENTENCES = [
    'Haluaisin kiittää {s} hänen nopeasta toiminnastaan, kun hän huolehti kaatuneen puun poistamisesta tieltämme, mikä paransi merkittävästi alueemme turvallisuutta. ',
    '{s} ansaitsee kiitokset siitä, kuinka hienosti hän on pitänyt huolta puistojemme kunnossapidosta, mikä tekee niistä viihtyisiä ja kauniita paikkoja kaikille. ',
    'Kiitän {s} hänen tehokkaasta tavastaan käsitellä melusaasteeseen liittyviä valituksia, mikä on parantanut monien asukkaiden elämänlaatua. ',
    'Olen iloinen nähdessäni, että {s} on ottanut toiveemme huomioon ja järjestänyt lisää leikkialueita lapsille, mikä on tuonut iloa ja aktiivisuutta yhteisöömme. ',
    'On mahtavaa, että {s} on huomioinut pyyntömme tarkistaa katuvalojemme toimivuus, ja nyt kaduillamme on turvallista liikkua myös pimeän aikaan. ',
    '{s} on tehnyt loistavaa työtä järjestäessään kesätapahtumat kaupungin aukiolla, joka on tuonut yhteen eri ikäisiä asukkaita ja luonut yhteisöllisyyden tunnetta. ',
    'On hienoa, että {s} on ottanut huomioon palautteemme parkkipaikkojen jaon osalta ja tehnyt muutoksia, jotka tuntuvat reilummilta kaikille. ',
    '{s} on ollut suureksi avuksi ja antanut arvokkaita neuvoja katujuhlan järjestämisessä, mikä on mahdollistanut onnistuneen ja ikimuistoisen tapahtuman. ',
    '{s} on tehnyt kiitettävää työtä kaupungin roskaongelman ratkaisemisessa, ja hänen toimensa ovat tehneet ympäristöstämme puhtaamman ja viihtyisämmän. ',
    'Olen myös kiitollinen {s}, joka on omistautunut vihreämpien vaihtoehtojen edistämiselle julkisessa liikenteessä, mikä on parantanut kaupunkimme ilmanlaatua ja edistänyt kestävää kehitystä.',
    'Kiitokset kaupungin työntekijöille, jotka huolehtivat Esplanadin puiston kauniista kukkaistutuksista, ne piristävät päivittäistä kävelyäni {s}. ',
    'On hienoa nähdä, että {s} joulukadun valot saavat aina niin paljon iloa ja lämpöä pimeään talviaikaan. ',
    'Kaupunki on tehnyt upeaa työtä luomalla kävely- ja pyöräteitä {s}, mikä on rohkaissut ihmisiä liikkumaan enemmän. ',
    'Suuri kiitos nopeasta reagoinnista ja {s} kuoppien paikkaamisesta; pyöräily on nyt paljon turvallisempaa. ',
    'Olen todella kiitollinen, että {s} leikkipuisto on kunnostettu, ja se tarjoaa nyt lapsille vielä enemmän hauskoja ja turvallisia leikkivälineitä. ',
    'Kaupungin vihertyöntekijät ansaitsevat kiitosta puhtaanapitoon liittyvästä työstä, jota he tekevät {s} varrella, se pitää alueen siistinä ja viihtyisänä. ',
    'On ilahduttavaa, että {s} varrella istutetut uudet puut tarjoavat varjoa kesäpäivinä ja lisäävät alueen viihtyisyyttä. ',
    'Kaupungin nopea toiminta vandaalismin korjaamisessa {s} leikkipaikalla on osoitus siitä, että he välittävät yhteisön viihtyisyydestä. ',
    'Kiitos, että {s} katutyöt saatiin valmiiksi etuajassa, mikä on vähentänyt huomattavasti alueella asuvien arjen haittoja. ',
    'Kruununhaan uudet liikennemerkit ja suojatie {s} ovat tehneet jalankulkijoiden liikkumisesta turvallisempaa ja sujuvampaa.',
    '{s} kirjaston uudet tilat ovat ihastuttava lisä alueen kulttuuritarjontaan, ja niiden monipuoliset tapahtumat ja työpajat ovat saaneet suuren suosion asukkaiden keskuudessa. ',
    '{s} ranta-alueiden siisteyteen ja ylläpitoon panostaminen on tuonut alueelle lisää ulkoilijoita ja parantanut kaikkien viihtyvyyttä. ',
    'Minusta {s} ympäristössä toteutetut viheralueiden kunnostustyöt ovat elävöittäneet koko aluetta ja houkutelleet paikalle niin lintubongareita kuin sunnuntaikävelijöitäkin. ',
    'Tänä kesänä avatut {s} uudet pyörätiet ovat tehneet saaresta entistä houkuttelevamman kohteen pyöräilijöille ja korostaneet alueen luonnonkauneutta. ',
    'Uskon enttä kaikki {s} asukkaat ovat olleet erityisen tyytyväisiä alueen katujen uudistuksiin, jotka ovat parantaneet liikenneturvallisuutta ja lisänneet viihtyisyyttä kävelyreiteillä.',
    'Kaupungin järjestämät ilmaiset konsertit puistossa ovat tuoneet yhteen eri-ikäisiä ihmisiä ja lisänneet yhteisön yhteenkuuluvuuden tunnetta.',
    'On hienoa, että kirjastot tarjoavat nyt entistä enemmän työtiloja etätyötä tekeville, mikä on edistänyt monimuotoisen työkulttuurin kehittymistä.',
    'Kiitokset kaupungille nopeasta toiminnasta poistettaessa graffiteja julkisilta seiniltä, se on pitänyt ympäristömme siistinä ja viihtyisänä.',
    'Puistojen leikkipaikkojen säännöllinen huolto ja turvallisuuden tarkastukset ovat olleet todella tärkeitä lasten turvallisuuden kannalta.',
    'Kaupungin viimeaikainen panostus pyöräteiden kunnossapitoon on tehnyt pyöräilystä miellyttävämmän ja turvallisemman vaihtoehdon kaupunkiliikenteessä.'
]

EVALUATION_VALUES = [
    'Martti',
    'SEPPO TOIVONEN',
    'Kalle',
    'Virtanen',
    'Matti Mäkelä',
    'Johanna Virkkunen',
    'Laaksonen',
    'Anna Lehtinen',
    'Sanna Holmström',
    'Sofia Virtaselle',
    'Mannerheimintiellä',
    'Aleksanterinkadun',
    'Pitkänsillanrantaan',
    'Helsinginkadun',
    'Vuorikadun',
    'Tehtaankadun',
    'Bulevardin',
    'Unioninkadun',
    'Kapteeninkadun',
    'Pohjoisesplanadilla',
    'Kallion',
    'Vuosaaren',
    'Töölönlahden',
    'lauttasaaren'
    'punavuoren',
    None,
    None,
    None,
    None,
    None,
]

EVALUATION_LABELS = [
    NAME_ENTITY,
    NAME_ENTITY,
    NAME_ENTITY,
    NAME_ENTITY,
    NAME_ENTITY,
    NAME_ENTITY,
    NAME_ENTITY,
    NAME_ENTITY,
    NAME_ENTITY,
    NAME_ENTITY,
    STREET_ENTITY,
    STREET_ENTITY,
    STREET_ENTITY,
    STREET_ENTITY,
    STREET_ENTITY,
    STREET_ENTITY,
    STREET_ENTITY,
    STREET_ENTITY,
    STREET_ENTITY,
    STREET_ENTITY,
    AREA_ENTITY,
    AREA_ENTITY,
    AREA_ENTITY,
    AREA_ENTITY,
    AREA_ENTITY,
    None,
    None,
    None,
    None,
    None,
]




print("Formatting training data into spacy examples...")

for s in NAME_LIST:
    sentence, start, end = generate_sentence(s, SENTENCES_NAME)
    doc = nlp(sentence)
    entities = [[start, end, NAME_ENTITY]]
    example: Example = Example.from_dict(doc, {"entities": entities})
    TRAIN_DATA.append(example)

street_suffixes = ['llä', 'lle']
for s in STREET_LIST:
    s = s.lower()
    if not ' ' in s and any(x in s for x in ['katu, tie, polku']):
        # use full set only with traditional street names
        sentence, start, end = generate_sentence(s, SENTENCES_STREETS)
    else:
        sentence, start, end = generate_sentence(s, SENTENCES_STREETS[:11])

    parts = s.split(' ')
    entities = []
    if len(parts) > 1:
        i = start
        for p in parts:
            j = i + len(p)
            entities.append([i, j, STREET_ENTITY])
            i = j + 1
    else:
        entities = [[start, end, STREET_ENTITY]]

    doc = nlp(sentence)

    example: Example = Example.from_dict(doc, {"text": sentence, "entities": entities})
    TRAIN_DATA.append(example)

for s in AREA_LIST:
    sentence, start, end = generate_sentence(s.lower(), SENTENCES_AREAS)
    doc = nlp(sentence)
    entities = [[start, end, AREA_ENTITY]]
    example: Example = Example.from_dict(doc, {"entities": entities})
    TRAIN_DATA.append(example)


# Add here example sentences that are used to teach not anonymizable sentences
FALSE_POSITIVES = [
    'Aura on naisen nimi mutta tässä yhteydessä viittaan kalustoon joka poistaa lunta kadulta.'
    'Maunulan Majalla maistuu mehu ja pulla'
    'Kruunuhaan Meritullintiellä autot ovat siististi parkissa'
    'Viikissä Mannerheimintiellä on katutyö'
    'Malmilla Kissantiellä kaikki hyvin'
    'Vaaditaan ed. mainittujen katujen lakaisua, etenkin Einarinkuja',
    'Flöitti dianan kujan bussipysäkillä on lasinsiruja',
    'Kekkolan tien risteyksessä on siili',
    'Kustaa Aadofin kadun varrella',
    'Mattilan ja rudolfinkadun risteyksessä',
    'Uunonkujalla puussa on lehtiä',
    'Eilen 12.3.45 tapahtui kevään ensimmäinen päivä, joka oli tiistai',
    'Terveisin: Mansikkanimen koulun henkilökunta',
    'Mahtilan Koulu',
    'Aura ajoi seinääni.',
    'Aura-auton kuski käyttää sinistä hattua',
    'Vaaditaan ed. mainittujen katujen aurausta! '
    'Ala-asteen pihassa on upea PUU'
    'Minusta tontti 38161/3 on kaunis.',
    'EU-kansalaisen kotimaa on eu-alueella.',
    'Helsingfors eli Helsinki, kuten täällä pää-hesassa sanomme.',
    'Erinomainen idea laittaa välille Jätkäsaari-Bulevardi omenoita.',
    'Pe21.1.22. kadut lakaistiin.',
    'Tämä ylilääkäri OLI ENSIMMÄINEN TERVEYSKESKUSLÄÄKÄRI perjantaina',
    'Esim tämä on esimerkki.',
    'Tiistaina me syödään esim keittoa',
    'olisin vapaa esim huomenna.',
    'Herjaa tämä tietokone että väärä tunnus ja salasana',
    'VIELÄKÄÄN EN OLE SAANUT LUMIKASAA POIS IKKUNAN ALTA.',
    'Bostads Ab Munksnäsallén no 1:n edessä oleva koivu kaipaa pikkulintuja.',
    '70-vuotiaita ja sitä vanhempia rokotetaan Pfizerin rokotteella.',
    'Bussimatkan hinta 16,00 € on aivan liian suuri',
    'Helsingin paras jäätelö on kylmää.',
    'Kaipaan Helsingissä sitä tunnelmaa mikä on Espoossa.',
    'Helsingissä voisi olla enemmän rapuja',
    'Muutettuani helsinkiin, ajattelin että voisin asua muutaman vuoden.',
    'Helsinkiläisin asia on on olla helsingistä kotoisin.',
    'Helsinkiläiset koirat ovat nisäkkäitä.',
    'Voisin vielä keksiä lauseen, jossa mainitaan helsingin sana, mutta en keksi enempää.'
]

for sentence in FALSE_POSITIVES:
    doc = nlp(sentence)
    c = 0
    entities = []
    for s in sentence.split(' '):
        start = c
        end = start + len(s)
        entities.append([start, end, 'O'])
        c = end + 1

    example: Example = Example.from_dict(doc, {"entities": entities})
    TRAIN_DATA.append(example)

EVAL_DATA = []
for i in range(0, len(EVALUATION_SENTENCES)-1):
    sentence = EVALUATION_SENTENCES[i]
    s = EVALUATION_VALUES[i]
    label = EVALUATION_LABELS[i]
    if s:
        sentence, start, end = generate_evaluation_sentence(s.lower(), sentence)
        doc = nlp(sentence)
        entities = [[start, end, label]]
        example: Example = Example.from_dict(doc, {"entities": entities})
    else:
        example: Example = Example.from_dict(nlp(sentence), {"entities": []})
    TRAIN_DATA.append(example)

# Heikki on kissa
# {text: 'Heikki on kissa', entities=[[14, 17, 'ELÄIN']]}

def train(training_iterations=1, score_threshold=0, verbose=False):

    print("Train Spacy NER model with names. Iterations = {i}, score threshold = {s}".format(i=training_iterations, s=score_threshold))
    print("using {l} sentences".format(l=len(TRAIN_DATA)))
    print("- containing  {n} examples with generated names.".format(n=len(NAME_LIST)))
    print("- containing  {n} examples with streets.".format(n=len(STREET_LIST)))
    print("- containing  {n} examples with areas.".format(n=len(AREA_LIST)))
    print("- containing  {n} examples with random sentences.".format(n=len(FALSE_POSITIVES)))

    # NER training
    def evaluate(nlp, data, verbose=False):
        scores = nlp.evaluate(data)
        if "ents_p" in scores:
            precision = scores["ents_p"]
            recall = scores["ents_r"]
            f1_score = scores["ents_f"]
            if verbose:
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1_score}")
        if "ents_per_type" in scores:
            per_type = scores["ents_per_type"]
            if verbose:
                for entity_type, metrics in per_type.items():
                    print(f"{entity_type}: Precision: {metrics['p']}, Recall: {metrics['r']}, F1: {metrics['f']}")
        else:
            if verbose:
                print("No entity types found", scores)
        return scores

    if exec_ruler:
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", after="ner", config={"phrase_matcher_attr": "LOWER", "overwrite_ents": True})
        else:
            ruler = nlp.get_pipe("entity_ruler")
        ruler.add_patterns(build_patterns(_PRODUCTS, 'PRODUCT'))
        ruler.add_patterns(build_patterns(_STREETS, STREET_ENTITY))
        ruler.add_patterns(build_patterns(_AREAS, AREA_ENTITY))
        ruler.add_patterns(build_patterns(_ORGANIZATIONS, 'ORG'))
        ruler.add_patterns(build_patterns(_SKIP, 'O'))
        print("Evaluate after entity ruler update")
        evaluate(nlp, EVAL_DATA, verbose=verbose)

    if exec_ner:
        n_iter = training_iterations

        score = evaluate(nlp, EVAL_DATA)
        print(f"Scores before update: {score}")

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.resume_training()
            for i in range(n_iter):  # Number of training iterations
                # Batch up the examples using spaCy's minibatch
                random.shuffle(TRAIN_DATA)
                losses = {}
                # Update the model with the new examples
                c = 0
                if verbose:
                    print(f"{i} / {c}: {losses}")
                batches = minibatch(TRAIN_DATA, size=compounding(16., 32., 1.001))
                for batch in batches:
                    c += 1
                    nlp.update(batch, drop=0.5, losses=losses, sgd=optimizer)

        for p in other_pipes:
            nlp.enable_pipe(p)
    test_score = 0
    eval_results = evaluate_nlp(nlp)

    if exec_test:
        print("\nAfter training test coverage is now: ")
        test_score = run_test(amount=100)
        print(f"\nScores after entity ruler update:")
        scores = evaluate(nlp, EVAL_DATA)
        print(scores)

    if save_model:
        if test_score > score_threshold:
            print(f"Saving model with score {test_score} to {target_path}")
            nlp.to_disk(target_path)
        else:
            print(f"Model test score {test_score} is below threshold {score_threshold}, not saving model.")
    return test_score, eval_results


if __name__ == "__main__":
    iterations = [1]
    test_score = 0
    highest_score = 0
    results = []
    timestamp = datetime.datetime.now().strftime('%Y.%m.%d %H:%M')
    with open(f"training_{timestamp}.txt", "a") as f:
        f.write(f"NAMES: {NAMES_TEST_DATA_SIZE} ")
        f.write(f"STREETS: {STREETS_TEST_DATA_SIZE} ")
        f.write(f"AREAS: {AREAS_TEST_DATA_SIZE} \n")
    for i in iterations:
        print(f"\n\n\nTraining with {i} iterations\n\n\n")
        test_score, eval_results = train(training_iterations=i, score_threshold=highest_score)
        print(f"Test score after {i} iterations: {test_score}")
        if test_score > highest_score:
            highest_score = test_score
        # log to file
        with open("training.log", "a") as f:
            f.write(f"{datetime.datetime.now().strftime('%Y.%m.%d %H:%M')}: Training with {i} iterations. Test score: {test_score}. Model {base_model}.  Augmented training data: Names {NAMES_TEST_DATA_SIZE}, Streets {STREETS_TEST_DATA_SIZE}, Areas {AREAS_TEST_DATA_SIZE}\n")
        stats = "Iterations " + str(i) + ". Test_score: " + str(test_score) + "\n" + eval_results + "\n\n\n"
        results.append(stats)
        # Full report
        with open(f"training_{timestamp}.txt", "a") as f:
            f.write("Training run: " + datetime.datetime.now().strftime('%Y.%m.%d %H:%M') + "\n")
            f.write(stats)

    # finally print the results
    for r in results:
        print(r)
