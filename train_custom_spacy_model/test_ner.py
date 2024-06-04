import spacy

model = "../custom_spacy_model/fi_datahel_spacy-0.0.1"

nlp = spacy.load(model)

text_fi = "Jaakko söi puuroa aurinkoisena päivänä Helsingissä vartiokylän puistossa, osoitteessa ylerminkuja 6. " \
           "Jaakko Parantainen käy mielelllään kalassa Helsingin itäpuolella (kaukana) yhdeksän kilometrin päässä sijaitsevassa kallion kaupunginosassa, " \
           "vaikka asuukin mannerheimintien varrella. Osoitteessa Liisankatu 12 U 2. Malmin kalatorilta ostetut Silakat ovat Jaakon herkkua."

doc = nlp(text_fi)
for token in doc:
    print(token.text, token.pos_, token.dep_, token.lemma_, token.ent_type_)

print([(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents])

