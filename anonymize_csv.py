import argparse
import csv
import sys
import time

from text_anonymizer import TextAnonymizer
from text_anonymizer.default_settings import RECOGNIZER_CONFIGURATION_ALL

parser = argparse.ArgumentParser(description='Anonymize csv file', epilog="Example: python anonymize_csv.py file_in.csv file_out.csv --column_name=text")
parser.add_argument('source_file', type=str, help='CSV-file to be anonymized')
parser.add_argument('target_file', type=str, help='Name or path of (anonymized) destination file.')

parser.add_argument('--column_name', type=str, help='Name (header) of anonymized column. Default: first column')
parser.add_argument('--column_index', type=str, help='Index(es) (starting from 0) of anonymized column(s). Default: 0 (first column). Select multiple columns by separating column indexes by comma.')
parser.add_argument('--header', type=str, help='CSV file contains header. Default: True')
parser.add_argument('--delimiter', type=str, help='CSV-file delimiter. Default: ;')
parser.add_argument('--quotechar', type=str, help='CSV quote character: SINGLE, DOUBLE.  Default: none')
parser.add_argument('--quotemode', type=str, help='CSV quoting mode: NONE, NON_NUMERIC, MINIMAL.  Default: NONE')
parser.add_argument('--languages', type=str, help='Selected languages (comma separated). Default: fi,en')
parser.add_argument('--encoding', type=str, help='Source encoding. Default: UTF-8')
parser.add_argument('--debug', type=str, help='Toggle debug logging. Default: False')
parser.add_argument('--recognizers', type=str, help=f'Override active recognizers. Available options: {", ".join(RECOGNIZER_CONFIGURATION_ALL)}')

delimiter = ';'
quotechar = ''
csv_file = None
column_name = None
column_names = []
column_index = 0
column_indexes = []
header = True
languages = ['fi']
debug = False
start_time = time.time()
source_encoding = 'UTF-8'
recognizers=None

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

if args.source_file:
    csv_file = args.source_file
if args.target_file:
    target_file = args.target_file
if args.column_name:
    if ',' in args.column_name:
        column_names = args.column_name.split(',')
    else:
        column_name = args.column_name
if args.column_index:
    if ',' in args.column_index:
        column_indexes = list(map(int, args.column_index.split(',')))
    else:
        column_index = int(args.column_index)
if args.header and "false" == args.header.lower():
    header = False

if args.quotechar:
    q = args.quotechar
    if q == 'SINGLE':
        quotechar = '\''
    elif q == 'DOUBLE':
        quotechar = '\"'
    else:
        quotechar = None
if args.delimiter:
    delimiter = args.delimiter
if args.languages:
    languages = args.languages.split(',')
if args.encoding:
    source_encoding = args.encoding
if args.debug and "true" == args.debug.lower():
    debug = True
if args.recognizers:
    recognizers = args.recognizers.split(',')


quoting = csv.QUOTE_NONE
if quotechar:
    quoting = csv.QUOTE_MINIMAL

if args.quotemode:
    q = args.quotemode
    if q == 'NONE':
        quoting = csv.QUOTE_NONE
    elif q == 'NON_NUMERIC':
        quoting = csv.QUOTE_NONNUMERIC
    elif q == 'MINIMAL':
        quoting = csv.QUOTE_MINIMAL
    else:
        quoting = csv.QUOTE_NONE



print("Anonymizing file: {i}. ".format(i=csv_file))
if debug:
    print("")
    print("Parameters:")
    print("- Source file: {s}".format(s=csv_file))
    print("- Target file: {s}".format(s=target_file))
    print("- Anonymized column name: {s}".format(s=column_name))
    if len(column_indexes) > 0:
        print("- Anonymized column indexes: {s}".format(s=column_indexes))
    else:
        print("- Anonymized column index: {s}".format(s=column_index))
    print("- File contains header: {s}".format(s=header))
    print("- CSV quote char: {s}".format(s=quotechar))
    print("- CSV quoting mode: {s}".format(s=quoting))
    print("- Quoting mode: {s}".format(s=quoting))
    print("- CSV delimiter: {s}".format(s=delimiter))
    print("- Encoding: {s}".format(s=source_encoding))
    print("")

text_anonymizer = TextAnonymizer(languages=languages, debug_mode=debug, recognizer_configuration=recognizers)
statistics_list = []
details_list = []

if csv_file:
    with open(target_file, 'w', newline='', encoding=source_encoding) as outfile:
        csv_writer = csv.writer(outfile, delimiter=delimiter, quotechar=quotechar, quoting=quoting)
        with open(csv_file, newline='', encoding=source_encoding) as in_file:
            try:
                csv_reader = csv.reader(in_file, delimiter=delimiter, quotechar=quotechar, quoting=quoting, strict=False)
                line_count = 0
                header_row = None
                anonymized = None
                if len(column_indexes) == 0 and column_index:
                    column_indexes = [column_index]

                if len(column_names) == 0 and column_name:
                    column_names = [column_name]

                if not column_indexes and not column_names:
                    print("Error: you need to define columns to be anonymized")
                    exit(-1)
                for row in csv_reader:
                    round_time = time.time()
                    if line_count == 0 and header:
                        if debug:
                            print("First row: ", row)
                        line_count += 1
                        header_row = row
                        if len(column_names) > 0:
                            for column_name in column_names:
                                i = row.index(column_name)
                                column_indexes.append(i)
                        print("Selected columns: indexes={}\n".format(column_indexes))
                        csv_writer.writerow(row)
                    else:
                        writerow = row.copy()
                        for i in column_indexes:
                            if row and len(row) > i:
                                text = row[i]
                                if text:    # can be empty
                                    anonymized = text_anonymizer.anonymize(text)
                                    if anonymized.statistics:
                                        statistics_list.append(anonymized.statistics)
                                    if anonymized.details:
                                        details_list.append(anonymized.details)
                                    writerow[i] = anonymized.anonymized_text
                                if debug:
                                    anonymized_text = anonymized.anonymized_text if anonymized else ""
                                    print(f"Original:\n{text}\n-->\nAnonymized:\n{anonymized_text}\n---\n")
                        csv_writer.writerow(writerow)
                        line_count += 1
            except ValueError as e:
                print("Failed to read csv file. Please check file format and parameters. Use --debug=True option for more information.")
                print(e)


print("\nFinished. Wrote anonymized version to: "+target_file)
print("--- Processing ready in  %s seconds ---" % round(time.time() - start_time))

combined_stats = text_anonymizer.combine_statistics(statistics_list)
combined_details = text_anonymizer.combine_details(details_list)
print("Statistics: ", combined_stats)
if debug:
    print("Details: ", combined_details)