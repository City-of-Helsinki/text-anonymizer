import datetime
from collections import defaultdict

from spacy import load
from spacy.training import Example
from tabulate import tabulate

from model_version import FINETUNED_MODEL_VERSION

'''
This script evaluates the trained model with a set of sentences.
The evaluation data consists of sentences with single entity in each sentence.
The entity type is provided in the evaluation data.
The script evaluates the model with the evaluation data and prints the results.
'''

VERBOSE_ON = False

# Entity labels that we consider equivalent for locations (streets/areas)
EQUIVALENT_LOCATION_LABELS = {('LOC', 'GPE'), ('GPE', 'LOC')}


def _labels_match(expected_label: str, predicted_label: str) -> bool:
    """Return True if predicted_label is acceptable for expected_label.

    Treat LOC and GPE as equivalent for evaluation purposes so that
    streets predicted as GPE (or areas predicted as LOC) are not counted
    as failures. Base model sometimes classifies street names as GPE and this
    is taken care in recognizers logic.
    """
    if expected_label == predicted_label:
        return True
    if (expected_label, predicted_label) in EQUIVALENT_LOCATION_LABELS:
        return True
    return False


def evaluate_nlp(nlp=None):
    if not nlp:
        # Load trained model
        model_path = f"../custom_spacy_model/{FINETUNED_MODEL_VERSION}"
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
    TIME = 'TIME'
    FAC = 'FAC'
    EVENT = 'EVENT'
    PRODUCT = 'PRODUCT'
    NORP = 'NORP'
    WORK_OF_ART = 'WORK_OF_ART'
    QUANTITY = 'QUANTITY'
    MONEY = 'MONEY'
    PERCENT = 'PERCENT'



    # Evaluation data
    # We use these sentences to evaluate trained model.
    # Some are generated with large language model and some are quoted from wikipedia
    # Feel free to add sentences, but make sure that there is single entity in each sentence
    # If script fails, it is likely that spacy already detects that entity as different type or it is in different position.
    # Eg. 2022 -> Vuonna 2022

    sentence_tuples = [
        ("Kaupungin puutarhuri {} ilmaisi viime viikon lehdessä, että kaupungin viheralueiden hoito on parantunut merkittävästi.", "Liinus Järvenpää", PERSON),
        ("Kiitos, edustaja {}, uudesta pyörätiestä - se on parantanut liikkumistani kaupungissa.", "Gideon Lehti", PERSON),
        ("Toivon, että kaupunginvaltuutettu {} ottaa huomioon meidän alueen koulujen tarpeet tulevassa budjetissa.", "Hirvonen", PERSON),
        ("{} kirjoitti lehdessä, että kaupungin puistojen siisteys on parantunut huomattavasti viime kuukausina.", "Pirjo Siekkinen", PERSON),
        ("Tilaisuuden aluksi {} kiitti kaupungin työntekijöitä erinomaisesta työstä kaupungin siisteyden parantamiseksi.", "Alan Paavilainen", PERSON),
        ("{} ehdotti kaupunginvaltuustossa, että kaupunki investoisi lisää resursseja julkisten tilojen siisteyteen.", "Sven Karlsson", PERSON),
        ("Remontti {} 5 kohdalla kestää viikkoja.", "Fabianinkatu", LOC),
        ("Havaittu vakava vaurio {} kohdalla, joka voi aiheuttaa vaaratilanteita.", "Aleksanterinkatu", LOC),
        ("{} edessä oleva tie on erittäin huonossa kunnossa ja tarvitsee pikaisen korjauksen.", "Erottajankatu", LOC),
        ("Olen erittäin tyytymätön keskusta-aluueen talvikunnossapitoon, erityisesti {} osalta.", "liisankadun", LOC),
        ("Kaupungininsinööri {} ansaitsee kiitoksen panoksestaan kaupungin it-koulutusjärjestelmän kehittämisessä.", "Aukusti Lehtonen", PERSON),
        ("Toivon, että {} kunnossapitoon panostettaisiin enemmän, sillä liikennemäärät ovat kasvussa.", "pohjoisesplanadin", LOC),
        ("Kiitos kaupungin kesätyöntekijöille, jotka ovat pitäneet {} paikat siistinä ja istutukset kauniina.", "fredrikinkadulla", LOC),
        ("Tiedoksi että {} kohdalla katuvalo on ollut pimeänä jo usean viikon ajan.", "kalevankatu", LOC),
        ("Kaupunginvaltuutettu {} lupasi vaalikampanjassaan panostaa kaupungin puistojen viihtyisyyteen ja siisteyteen.", "Sisu Asikainen", PERSON),
        ("Pyydän kiinnittämään huomiota {} kadun liikennejärjestelyihin, jotka aiheuttavat turvallisuusriskejä.", "mechelininkadun", LOC),
        ("Haluaisin kiittää lakaisukoneen kuljettajia, jotka ovat tänä keväänä huolehtineet {} kunnossapidosta erinomaisella tavalla.", "kaisaniemenkadun", LOC),
        ("Toivon, että kaupunki panostaisi enemmän resursseja {} valaistuksen parantamiseen.", "bulevardin", LOC),
        ("{} esiintyi viime viikolla kaupunki ja kirja tapahtumassa.", "Jukka-Pekka Halonen", PERSON),
        ("Kiitos {}, että olet kuunnellut asukkaiden toiveita ja parantanut kaupungin viheralueiden hoitoa.", "Vilppu Meriläinen", PERSON),
        ("En voi käsittää miksi {} on niin huonossa kunnossa, että se aiheuttaa jo vahinkoja autoille.", "unioninkatu", LOC),
        ("Kiitos kaupungin työntekijöistä, jotka hoitavat {} kunnossapidosta.", "Katariinankatu", LOC),
        ("Toivoisin että {} olisi paremmassa kunnossa liikkumisen parantamiseksi.", "Hämeentie", LOC),
        ("Haluaisin kiittää kaupunginvaltuutettu {} siitä, että hän on nostanut esiin alueemme lähiökoulujen tarpeet.", "Tristan Lindströmiä", PERSON),
        ("Valitettavasti {} roskakorit ovat jatkuvasti täynnä, mikä aiheuttaa haittaa asukkaille.", "Yrjölänkujan", LOC),
        ("Edustaja {} lupasi vaalipuheessaan panostaa kaupungin koulutusjärjestelmän laadun parantamiseen.", "Räty", PERSON),
        ("Toivon, että kaupunginvaltuutettu {} ottaisi vakavasti asukkaiden huolenaiheet alikulkujen turvallisuudesta.","Olli-Pekka Heinonen", PERSON),
        ("Kiitos kaupungin työntekijöille, jotka ovat pitäneet {} kadun puhtaina ja hoidettuina.", "simonkadun", LOC),
        ("Olemme erittäin pettyneitä siihen, että {} jalkakäytävä on jatkuvasti huonossa kunnossa.", "soidinkujan", LOC),
        ("Toivoisin että {} olisi kunnossa.", "Kaisaniemenkatu", LOC),
        ("Kampus sijaitsee Helsingin ydinkeskustassa Kruununhaan ja Kluuvin kaupunginosissa {} molemmin puolin", "Unioninkadun", LOC),
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
        ("Jääkiekossa Helsingin suosituimmat ja menestyneimmät joukkueet ovat SM-liigajoukkue {} sekä nykyisin Mestiksessä pelaava Jokerit.", "HIFK", ORG),
        ("Helsingissä on kymmeniä elokuvasaleja, joista suurin on 635-paikkainen Tennispalatsin {} sali.", "ISENSE", O),
        ("Veneilijöitä varten kaupungin rannoilla on {} laituripaikkaa.", "noin 12 000", CARDINAL),
        ("Kalasatama (ruots. Fiskehamnen) on {} kaupunginosan osa-alue Helsingin itäisessä kantakaupungissa.", "Sörnäisten", LOC),
        ("Se käsittää suuren osan entistä {} sataman aluetta, jolle on alettu rakentaa uutta asunto- ja toimistoaluetta.", "Sörnäisten", LOC),
        ("Liikenteellisesti Kalasatama on keskeisellä paikalla, ja {} kaasukellojen pohjoispuolella sijaitseva liittymäalue on kantakaupungin solmupiste.", "Suvilahden", LOC),
        ("Alueelle rakennetaan myös uutta raitiotieyhteyttä {}.", "Pasilaan", LOC),
        ("Kalasatamasta on tulossa tiiviisti rakennettu alue – asukkaita sinne arvioidaan tulevan lopulta {}, suunnilleen yhtä paljon kuin Kalliossa.", "noin 20 000", CARDINAL),
        ("Lisäksi alueelle kaavaillaan työpaikkoja {} ihmiselle.", "noin 10 000", QUANTITY),
        ("Läheisestä Toukolan kaupunginosasta löytyy {}, jonka mukaan Kotisaaren leipomo on saanut nimensä.", "Kotisaarenkatu", LOC),
        ("Diakonia-ammattikorkeakoulun rakennus sijaitsee osoitteessa {} 2 Helsingin Kyläsaaressa.", "Kyläsaarenkuja", LOC),
        ("Kyläsaari esiintyy myös kirjailija {} romaanissa Rakastunut rampa.", "Joel Lehtosen", PERSON),
        ("Kyläsaaren ranta-alueella on osittain kuvattu ohjaaja {} elokuva Mies vailla menneisyyttä.", "Aki Kaurismäen", PERSON),
        ("Helsingissä toimii {} seudun liikenteen (HSL) järjestämänä kattava julkinen liikenne.", "Helsingin", GPE),
        ("Hotelliyöpymiset {} hotelleissa ovat lähes 13-kertaistuneet 50 vuodessa.", "Helsingin", GPE),
        ("{} päärautatieasema on Suomen matkustajaliikenteen keskus.", "Helsingin", GPE),
        ("Noin kolme kilometriä pohjoisemmalla {} rautatieasemalla rata haarautuu rantaratana länteen ja pääratana pohjoiseen.", "Pasilan", GPE),
        ("Kokous alkaa tarkalleen klo {} iltapäivällä.", "14.30", TIME),
        ("Viimeinen juna Helsinkiin lähtee yleensä {} aikaan.", "puolen yön", TIME),
        ("Uusi urheiluhalli {} valmistui Itäkeskukseen.", "Arena Center", ORG),
        ("Konsertti järjestetään paikassa {}.", "Messukeskus", ORG),
        ("Tapahtuma {} oli menestys.", "Flow Festival", ORG),
        ("Ostin {} puhelimen verkkokaupasta.", "Galaxy", PRODUCT),
        ("Tuotteen {} saatavuus on parempi nyt.", "iPhone", PRODUCT),
        ("Asun kerroksessa {}.", "seitsemännessä", CARDINAL),
        ("Hänelle on tulossa {} sija.", "kolmas", CARDINAL),
        ("Tilasin {} hiekkaa leikkikenttää varten.", "2", CARDINAL),
        ("Kaupunginpuutarha tilasi {} kukkasipuleita.", "5", CARDINAL),
        ("Rakennuksen hinta kaupungille oli {}.", "157890 euroa", MONEY),
        ("Budjetista varattiin {} korjaukseen.", "8000€", MONEY),
        ("Liikennemelun arvioidaan vähentyneen {}.", "30%", PERCENT),
    ]

    # Build examples from eval set
    all_eval_data = []
    failed_predictions = defaultdict(list)

    for sentence, entity_value, entity_label in sentence_tuples:
        formatted_sentence = sentence.format(entity_value)

        # Find entity position in formatted sentence
        try:
            start = formatted_sentence.index(entity_value)
            end = start + len(entity_value)
        except ValueError:
            print(f"Warning: Entity '{entity_value}' not found in sentence: {formatted_sentence}")
            continue

        # Special handling for O-labeled (non-entity) spans: we expect NO entity over this span
        if entity_label == O:
            doc = nlp(formatted_sentence)
            predicted_entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

            # Check if any predicted entity overlaps the expected O span
            overlapping_entities = [
                (text, label, s, e) for text, label, s, e in predicted_entities
                if not (e <= start or s >= end)
            ]

            if overlapping_entities:
                # Model incorrectly tagged a non-entity span
                failed_predictions[entity_label].append({
                    'sentence': formatted_sentence,
                    'expected': (entity_value, entity_label, start, end),
                    'predicted': overlapping_entities,
                    'note': 'Expected no entity for O label, but model predicted entities over the span'
                })

            # Do NOT add O examples to all_eval_data, since they are not true entity annotations
            continue

        # Normal entity example handling (PERSON, LOC, GPE, CARDINAL, etc.)
        entities = [[start, end, entity_label]]
        annotations = {"entities": entities}

        # Create example object
        example = Example.from_dict(nlp.make_doc(formatted_sentence), annotations)
        all_eval_data.append(example)

        # Check prediction for logging
        doc = nlp(formatted_sentence)
        expected_entity = (entity_value, entity_label, start, end)
        predicted_entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

        # Check if prediction matches expectation (with LOC/GPE equivalence)
        prediction_match = False
        matching_prediction = None

        # 1. Exact span + acceptable label
        for pred_text, pred_label, pred_start, pred_end in predicted_entities:
            if _labels_match(entity_label, pred_label) and pred_start == start and pred_end == end:
                prediction_match = True
                matching_prediction = [(pred_text, pred_label, pred_start, pred_end)]
                break

        # 2. If no exact match, check for combined overlapping entities with acceptable labels
        if not prediction_match:
            overlapping_entities = [
                (text, label, s, e) for text, label, s, e in predicted_entities
                if _labels_match(entity_label, label) and not (e <= start or s >= end)
            ]

            if overlapping_entities:
                covered_chars = set()
                for _, _, s, e in overlapping_entities:
                    covered_chars.update(range(s, e))

                expected_chars = set(range(start, end))
                intersection_size = len(expected_chars.intersection(covered_chars))
                expected_size = len(expected_chars)
                coverage_ratio = intersection_size / expected_size if expected_size > 0 else 0

                if coverage_ratio >= 0.9:
                    prediction_match = True
                    matching_prediction = overlapping_entities

        if not prediction_match:
            failed_predictions[entity_label].append({
                'sentence': formatted_sentence,
                'expected': expected_entity,
                'predicted': predicted_entities,
                'note': 'Entity not detected, wrong label, or insufficient overlap'
            })

    # Evaluation results
    eval_results = nlp.evaluate(all_eval_data)
    eval_results_data = eval_results['ents_per_type']

    # Info about the dataset
    entity_counts = {}
    for example in all_eval_data:
        for entity in example.reference.ents:
            entity_counts[entity.label_] = entity_counts.get(entity.label_, 0) + 1

    # Combine evaluation results and entity counts
    for key, value in eval_results_data.items():
        for key2, value2 in entity_counts.items():
            if key == key2:
                value['count'] = value2

    # Convert to table
    eval_results_table_data = [[key] + list(values.values()) for key, values in eval_results_data.items()]
    headers = ['Entity', 'precision', 'recall', 'f1-score', 'samples']
    eval_results_markdown_table = tabulate(eval_results_table_data, headers, tablefmt="pipe")

    # Print results
    print("\n\n### Evaluation results for model")
    print("\nEvaluation dataset consists of {} sample sentences.\n".format(len(all_eval_data)))
    print(f"\nDate: {datetime.datetime.now().strftime('%d.%m.%Y')}\n")
    print("\nEvaluation results: \n")
    print(eval_results_markdown_table)
    print("")

    # Print failed predictions
    if failed_predictions and VERBOSE_ON:
        print("\n### Failed Predictions\n")
        for entity_type, failures in failed_predictions.items():
            print(f"\n#### {entity_type} ({len(failures)} failures)\n")
            for idx, failure in enumerate(failures, 1):
                print(f"{idx}. Sentence: {failure['sentence']}")
                exp_text, exp_label, exp_start, exp_end = failure['expected']
                print(f"   Expected: '{exp_text}' as {exp_label} at position [{exp_start}:{exp_end}]")
                if failure['predicted']:
                    print(f"   Predicted:")
                    for pred_text, pred_label, pred_start, pred_end in failure['predicted']:
                        print(f"     - '{pred_text}' as {pred_label} at position [{pred_start}:{pred_end}]")
                else:
                    print(f"   Predicted: No entities detected")
                print()

    return eval_results_markdown_table


if __name__ == "__main__":
    evaluate_nlp()


# Date: 01.12.2025
#
#
# Evaluation results:
#
# | Entity   |   precision |   recall |   f1-score |   samples |
# |:---------|------------:|---------:|-----------:|----------:|
# | PERSON   |    0.527778 | 0.926829 |   0.672566 |        41 |
# | DATE     |    0.134615 | 0.466667 |   0.208955 |        15 |
# | ORG      |    0        | 0        |   0        |           |
# | LOC      |    0.5      | 0.741935 |   0.597403 |        31 |
# | CARDINAL |    0        | 0        |   0        |         4 |
# | GPE      |    0.136364 | 0.857143 |   0.235294 |         7 |
# | O        |    0        | 0        |   0        |         3 |
