import csv, json, sys
from collections import defaultdict

with open('data/full/confact/humc_cleaned_v2_stats.csv') as f:
    rows = list(csv.DictReader(f))

with open('data/full/confact/humc.json') as f:
    data = json.load(f)
doc_index = {doc['id']: doc for qi in data for doc in qi['documents']}

targets = ['www.refugeecouncil.org.uk','www.knbs.or.ke','www.dailypioneer.com',
           'ltoportal.ph','kippra.or.ke','www.justice.gov','www.coca-colacompany.com',
           'www.cambridge.org','www.ageuk.org.uk','dailytrust.com','www.thisdaylive.com',
           'www.thelallantop.com','www.ssa.gov','www.ojp.gov','www.indiabudget.gov.in',
           'www.freep.com','www.europarl.europa.eu','www.coca-cola.com','www.cbse.gov.in',
           'www.businessdailyafrica.com','www.amnesty.org.uk','worldpopulationreview.com',
           'theweek.com','sansad.in','prsindia.org','newsroom.heart.org','fivethirtyeight.com',
           'explore-education-statistics.service.gov.uk']

domain_docs = defaultdict(list)
for row in rows:
    if row['source'] in targets and row['method'] == 'generic':
        domain_docs[row['source']].append(row['doc_id'])

out = open('sample_domains_output.txt', 'w')

for domain in targets:
    docs = domain_docs[domain]
    out.write(f'\n{"="*60}\n')
    out.write(f'DOMAIN: {domain} ({len(docs)} docs)\n')
    for doc_id in docs:
        raw = doc_index[doc_id]['content']
        out.write(f'\n  [{doc_id}] len={len(raw)}\n')
        for start in [0, 600, 1200, 2000]:
            if start < len(raw):
                out.write(f'  [{start}:{start+400}]: {repr(raw[start:start+400])}\n')

out.close()
print("Done. Output written to sample_domains_output.txt")
