import spacy

from model_version import FINETUNED_MODEL_VERSION

model = f"../custom_spacy_model/{FINETUNED_MODEL_VERSION}"

nlp = spacy.load(model)

text_fi = """
Kokous alkaa tarkalleen klo 14.30 iltapäivällä.
Viimeinen juna Helsinkiin lähtee yleensä puolen yön aikaan.

Uusi urheiluhalli Arena Center valmistui viime vuonna Itäkeskukseen.
Konsertti järjestetään tänä vuonna Messukeskuksen päärakennuksessa.

Kesän suosituin tapahtuma on edelleen Flow Festival Suvilahdessa.
Kaupungin järjestämä Valon yö keräsi tuhansia osallistujia keskustaan.

Ostin eilen uuden puhelimen mallia Galaxy S24 Kampin liikkeestä.
Suosikkikahvini on tummapaahtoinen Juhla Mokka aamuisin.

Monet ruotsalaiset turistit vierailevat Helsingissä kesäisin.
Paikalla oli paljon saksalaisia opiskelijoita vaihto-ohjelman kautta.

Asun talon seitsemännessä kerroksessa, josta näkyy merelle.
Hän tuli maaliin kolmannella sijalla maratonilla.

Luin hiljattain romaanin Tuntematon sotilas ja pidin siitä paljon.
Suosikkielokuvani on edelleen Mies vailla menneisyyttä.

Tilasin varastoon kaksi kuutiota hiekkaa leikkikenttää varten.
Kaupunginpuutarha tilasi viisi säkillistä kukkasipuleita kevääksi.

Uuden pyörätien rakentaminen maksoi kaupungille noin 2,5 miljoonaa euroa.
Kunnan budjetista varattiin 800 000 euroa koulurakennusten korjaukseen.

Liikennemelun arvioidaan vähentyneen noin 30 prosenttia viime vuoden aikana.
Uuden joukkoliikennereitin myötä autoilun osuus on laskenut 15 prosenttia.
"""
doc = nlp(text_fi)
for token in doc:
    print(token.text, token.pos_, token.dep_, token.lemma_, token.ent_type_)

print([(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents])

