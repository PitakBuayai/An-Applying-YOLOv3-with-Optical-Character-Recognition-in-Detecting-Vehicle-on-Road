import csv

with open('example2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age', 'Gender'])
    writer.writerow(['Alex', '13', 'M'])
    writer.writerow(['Ripley', '32', 'F'])
