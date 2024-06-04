import argparse

from text_anonymizer import TextAnonymizer
def main():

    parser = argparse.ArgumentParser(description='Anonymize txt file', epilog="Example: python anonymize_txt.py file_in.txt file_out.txt --column_name=text")
    parser.add_argument('source_file', type=str, help='Text file to be anonymized')
    parser.add_argument('target_file', type=str, help='Name or path of (anonymized) destination file.')
    parser.add_argument('--debug', type=str, help='Toggle debug logging. Shows scores within labels. (True/False)')
    parser.add_argument('--languages', type=str, help='Selected languages (comma separated). Default: fi,en')
    parser.add_argument('--encoding', type=str, help='Source encoding. Default: UTF-8')
    parser.add_argument('--separator', type=str, help='String separator for newlines. Default: None')

    debug = False

    args = parser.parse_args()
    languages = ['fi', 'en']
    source_encoding = 'UTF-8'
    source_file = None
    separator = None

    if args.source_file:
        source_file = args.source_file
    if args.target_file:
        target_file = args.target_file
    if args.debug and "true" == args.debug.lower():
        debug = True
    if args.languages:
        languages = args.languages.split(',')
    if args.encoding:
        source_encoding = args.encoding
    if args.separator:
        separator = args.separator

    print("Anonymizing file: {i}. ".format(i=source_file))


    print("")
    print("Parameters:")
    print("- Source file: {s}".format(s=source_file))
    print("- Target file: {s}".format(s=target_file))
    print("- Debug mode: {s}".format(s=debug))
    print("")

    text_anonymizer = TextAnonymizer(languages=languages, debug_mode=debug)
    statistics = []
    details = []

    def anonymize(doc: [str]) -> (str, object, object):
        a = ' '.join(doc)
        if a:
            result = text_anonymizer.anonymize(a)
            return result.anonymized_text, result.statistics, result.details
        return None, None, None


    def prepare_raw_text(line):
        line = line.replace('\s+', ' ')
        return line


    if source_file:
        try:
            # use same encoding for source and target file
            with open(target_file, mode='w+', newline='', encoding=source_encoding) as outfile:
                with open(source_file, mode='r', newline='', encoding=source_encoding) as in_file:
                    lines = in_file.readlines()
                    doc = []
                    newline_counter = 0
                    line_counter = 0

                    for line in lines:
                        line_counter += 1
                        if line != '\n':
                            # remove double spaces etc
                            prepared = prepare_raw_text(line)
                            doc.append(prepared)
                        else:
                            newline_counter += 1

                        if newline_counter >= 2 or line_counter >= len(lines):
                            newline_counter = 0
                            anonymized, stats, detail = anonymize(doc)
                            if anonymized:
                                anonymized = ' '.join(anonymized.split())
                                if stats:
                                    statistics.append(stats)
                                if detail:
                                    details.append(detail)
                                if debug:
                                    if doc:
                                        print('>>> Original: ')
                                        print(''.join(doc))
                                        print('>>> Anonymized: ')
                                        print(anonymized)
                                        print('---')
                            doc = []
                            if anonymized:
                                outfile.write(anonymized)
                                if separator:
                                    outfile.write("\n{separator}\n")
        except Exception as e:
            print("Error: ", e)
            if 'codec' in str(e):
                print("Hint: Possibly invalid encoding. Please check the encoding of the source file. Use --encoding=... option to set the correct encoding.")
            exit(-1)
    combined_stats = text_anonymizer.combine_statistics(statistics)
    combined_details = text_anonymizer.combine_details(details)
    print("Statistics: ", combined_stats)
    if debug:
        print("Details: ", combined_details)
    print("\nFinished. Wrote anonymized version to: "+target_file)

if __name__ == "__main__":
    main()