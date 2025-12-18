"""
Testit EXAMPLE-tunnisteiden havaitsemiseen mukautetuilla regex-kuvioilla.

Tämä testi varmistaa, että kaikki EXAMPLE regex-kuviot example-profiilissa
tunnistavat oikein erilaisia EXAMPLE-sanan ja numeroiden yhdistelmiä.

REGEX-KUVIOIDEN OHJE:
=====================
Kenoviivat JSON regex-kuvioissa:
- JSONissa kenoviiva pitää escapoida: \\ tulee yhdeksi \ varsinaisessa regexissä
- \\b JSONissa = \b regexissä = sananraja
- \\d JSONissa = \d regexissä = numero [0-9]

Kuviosyntaksin pikaopas:
- \\b        Sananraja (estää osittaiset osumat kuten 'MYEXAMPLE' tai 'EXAMPLES')
- [0-9]     Vastaa mitä tahansa yksittäistä numeroa
- [A-Za-z]  Vastaa mitä tahansa yksittäistä kirjainta (iso tai pieni)
- +         Vastaa yhtä tai useampaa edeltävää elementtiä
- *         Vastaa nollaa tai useampaa edeltävää elementtiä
- {3}       Vastaa täsmälleen 3 edeltävää elementtiä
- {2,4}     Vastaa 2-4 edeltävää elementtiä
- (?:...)   Ei-kaappaava ryhmä vaihtoehdoille
- |         TAI-operaattori (käytetään ryhmien sisällä)

Esimerkkikuviot selitettynä:
- \\bEXAMPLE\\b           Vastaa täsmälleen sanaa "EXAMPLE"
- \\bEXAMPLE[0-9]+\\b     Vastaa "EXAMPLE" ja 1+ numeroa: EXAMPLE1, EXAMPLE987
- \\bEXAMPLE[0-9]*\\b     Vastaa "EXAMPLE" ja 0+ numeroa: EXAMPLE, EXAMPLE1
- \\bEXAMPLE[0-9]{3}\\b   Vastaa "EXAMPLE" ja täsmälleen 3 numeroa: EXAMPLE987
"""

import unittest
from text_anonymizer import TextAnonymizer


class TestCustomRegex(unittest.TestCase):
    """Testitapaukset EXAMPLE-tunnisteiden havaitsemiseen example-profiilissa."""

    def setUp(self):
        """Alusta tekstin anonymisoija ja testiparametrit."""
        self.label = "EXAMPLE"
        self.profile_name = "example"
        self.anonymizer = TextAnonymizer(debug_mode=False)

    # =========================================================================
    # Kuvio: exact_match - \\bEXAMPLE\\b
    # Vastaa vain täsmällistä sanaa "EXAMPLE" sananrajoilla
    # =========================================================================
    def test_exact_match_standalone(self):
        """Testaa täsmällisen EXAMPLE-sanan havaitseminen."""
        text = "EXAMPLE"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_exact_match_in_sentence(self):
        """Testaa EXAMPLE-sanan havaitseminen lauseessa."""
        text = "Tämä on EXAMPLE esimerkki tekstistä."
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn("EXAMPLE", result.details[self.label])

    def test_exact_match_not_partial_prefix(self):
        """Testaa ettei MYEXAMPLE vastaa (sananraja estää etuliitteen)."""
        text = "MYEXAMPLE"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # Ei pitäisi havaita EXAMPLE-sanaa MYEXAMPLE-sanan sisältä sananrajan takia
        if self.label in result.details:
            self.assertNotIn("EXAMPLE", result.details[self.label])

    def test_exact_match_not_partial_suffix(self):
        """Testaa ettei EXAMPLES vastaa täsmällistä kuviota."""
        text = "EXAMPLES"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        if self.label in result.details:
            self.assertNotIn("EXAMPLES", result.details[self.label])

    # =========================================================================
    # Kuvio: case_insensitive_variations - \\b[Ee][Xx][Aa][Mm][Pp][Ll][Ee]\\b
    # Vastaa EXAMPLE-sanaa missä tahansa kirjainkoossa
    # =========================================================================
    def test_case_insensitive_lowercase(self):
        """Testaa pienillä kirjaimilla kirjoitetun 'example' havaitseminen."""
        text = "example"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_case_insensitive_mixed_case(self):
        """Testaa sekakirjainkoon 'ExAmPlE' havaitseminen."""
        text = "ExAmPlE"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_case_insensitive_title_case(self):
        """Testaa otsikkokirjainkoon 'Example' havaitseminen."""
        text = "Example"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    # =========================================================================
    # Kuvio: word_with_numbers - \\bEXAMPLE[0-9]+\\b
    # Vastaa EXAMPLE ja yksi tai useampi numero perässä
    # =========================================================================
    def test_word_with_numbers_single_digit(self):
        """Testaa EXAMPLE ja yksi numero perässä."""
        text = "EXAMPLE1"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_word_with_numbers_multiple_digits(self):
        """Testaa EXAMPLE ja useita numeroita perässä."""
        text = "EXAMPLE987"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_word_with_numbers_many_digits(self):
        """Testaa EXAMPLE ja monta numeroa perässä."""
        text = "EXAMPLE999999"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_word_with_numbers_in_sentence(self):
        """Testaa EXAMPLE numeroilla lauseyhteydessä."""
        text = "Tarkista koodi EXAMPLE456 järjestelmästä."
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn("EXAMPLE456", result.details[self.label])

    # =========================================================================
    # Kuvio: word_with_fixed_digits - \\bEXAMPLE[0-9]{3}\\b
    # Vastaa EXAMPLE ja täsmälleen 3 numeroa
    # =========================================================================
    def test_fixed_digits_exact_three(self):
        """Testaa EXAMPLE ja täsmälleen 3 numeroa."""
        text = "EXAMPLE987"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_fixed_digits_too_few_not_matched(self):
        """Testaa ettei EXAMPLE56 (2 numeroa) vastaa fixed_digits-kuviota."""
        text = "EXAMPLE56"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # word_with_digit_range-kuvio vastaa tätä (2-4 numeroa)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_fixed_digits_too_many_not_matched_by_fixed(self):
        """Testaa että EXAMPLE9874 (4 numeroa) vastaa digit_range-kuviota."""
        text = "EXAMPLE9874"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    # =========================================================================
    # Kuvio: word_with_digit_range - \\bEXAMPLE[0-9]{2,4}\\b
    # Vastaa EXAMPLE ja 2-4 numeroa
    # =========================================================================
    def test_digit_range_two_digits(self):
        """Testaa EXAMPLE ja 2 numeroa."""
        text = "EXAMPLE56"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_digit_range_four_digits(self):
        """Testaa EXAMPLE ja 4 numeroa."""
        text = "EXAMPLE9874"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_digit_range_one_digit_no_match(self):
        """Testaa ettei EXAMPLE1 (1 numero) vastaa digit_range-kuviota."""
        text = "EXAMPLE1"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # word_with_numbers-kuvio vastaa tätä (1+ numeroa)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_digit_range_five_digits_no_match(self):
        """Testaa ettei EXAMPLE98745 (5 numeroa) vastaa digit_range-kuviota."""
        text = "EXAMPLE98745"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # word_with_numbers-kuvio vastaa (1+ numeroa)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    # =========================================================================
    # Kuvio: prefix_variations - \\b(?:TEST|PROD|DEV)_EXAMPLE[0-9]+\\b
    # Vastaa TEST_EXAMPLE, PROD_EXAMPLE, DEV_EXAMPLE ja numeroita
    # =========================================================================
    def test_prefix_test_example(self):
        """Testaa TEST_EXAMPLE numeroilla."""
        text = "TEST_EXAMPLE1"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_prefix_prod_example(self):
        """Testaa PROD_EXAMPLE numeroilla."""
        text = "PROD_EXAMPLE99"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_prefix_dev_example(self):
        """Testaa DEV_EXAMPLE numeroilla."""
        text = "DEV_EXAMPLE987"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_prefix_unknown_no_match(self):
        """Testaa ettei tuntematon etuliite STAGE_EXAMPLE vastaa."""
        text = "STAGE_EXAMPLE1"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # Ei pitäisi vastata prefix_variations-kuviota
        if self.label in result.details:
            self.assertNotIn(text, result.details[self.label])

    def test_prefix_in_sentence(self):
        """Testaa etuliitekuviot lauseyhteydessä."""
        text = "Julkaise TEST_EXAMPLE42 tuotantoon."
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn("TEST_EXAMPLE42", result.details[self.label])

    # =========================================================================
    # Kuvio: alphanumeric_suffix - \\bEXAMPLE[A-Za-z0-9]{3,6}\\b
    # Vastaa EXAMPLE ja 3-6 kirjain-numero-merkkiä
    # =========================================================================
    def test_alphanumeric_letters_only(self):
        """Testaa EXAMPLE ja pelkkiä kirjaimia perässä."""
        text = "EXAMPLEabc"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_alphanumeric_mixed(self):
        """Testaa EXAMPLE ja sekoitus kirjaimia ja numeroita."""
        text = "EXAMPLE1a2b"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_alphanumeric_uppercase(self):
        """Testaa EXAMPLE ja isoja kirjaimia perässä."""
        text = "EXAMPLEABC"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_alphanumeric_too_short(self):
        """Testaa ettei EXAMPLE ja 2 merkkiä vastaa alphanumeric-kuviota."""
        text = "EXAMPLEab"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # Ei pitäisi vastata alphanumeric_suffix-kuviota (vaatii 3-6)
        if self.label in result.details:
            self.assertNotIn(text, result.details[self.label])

    def test_alphanumeric_max_length(self):
        """Testaa EXAMPLE ja 6 kirjain-numero-merkkiä."""
        text = "EXAMPLEabcdef"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_alphanumeric_too_long(self):
        """Testaa ettei EXAMPLE ja 7+ merkkiä vastaa alphanumeric-kuviota."""
        text = "EXAMPLEabcdefg"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # Ei pitäisi vastata alphanumeric_suffix-kuviota (vaatii 3-6)
        if self.label in result.details:
            self.assertNotIn(text, result.details[self.label])

    # =========================================================================
    # Reunatapaukset ja useat osumat
    # =========================================================================
    def test_empty_text(self):
        """Testaa ettei tyhjä teksti tuota havaintoja."""
        text = ""
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertNotIn(self.label, result.details)

    def test_multiple_patterns_in_text(self):
        """Testaa useiden EXAMPLE-kuvioiden havaitseminen yhdessä tekstissä."""
        text = "Tarkista EXAMPLE1, EXAMPLE987 ja TEST_EXAMPLE1 järjestelmästä."
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        entities = result.details[self.label]
        # Tarkista että jokainen odotettu arvo löytyy listasta
        self.assertIn("EXAMPLE1", entities)
        self.assertIn("EXAMPLE987", entities)
        self.assertIn("TEST_EXAMPLE1", entities)

    def test_word_boundary_prevents_partial_match(self):
        """Testaa että sananrajat estävät osittaiset osumat."""
        text = "NOTEXAMPLE987HERE"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # Sananrajan pitäisi estää EXAMPLE987 osuman tämän merkkijonon sisällä
        if self.label in result.details:
            self.assertNotIn("EXAMPLE987", result.details[self.label])


if __name__ == "__main__":
    unittest.main()

