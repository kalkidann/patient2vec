"""
This file queries patient database and generates patient paragraph containing sequence of events.
"""

import psycopg2
import json


conn = psycopg2.connect(database="mimic", user="kalkidan", password = "kalkidan", host = "127.0.0.1", port = "5433")
print("Opened database successfully")


cur = conn.cursor()
cur.execute("SET search_path TO mimiciii")


# Dictionary for lab events
dl_dict = {}
query = "SELECT dl.ITEMID from d_labitems as dl ORDER BY dl.ITEMID"
cur.execute(query)
rows = cur.fetchall()
for i,row in enumerate(rows):
	dl_dict[row[0]] = i

with open('labevents_dict.json', 'w') as f:
	json.dump(dl_dict, f, indent=2)


# Labtests
query = "SELECT p.SUBJECT_ID, l.ITEMID, l.CHARTTIME from patients as p, labevents as l WHERE p.SUBJECT_ID=l.SUBJECT_ID ORDER BY p.SUBJECT_ID ASC, l.CHARTTIME ASC"
cur.execute(query)
rows = cur.fetchall()

lines_event = []
lines_time = []
line_e = ""
line_t = ""
pid = -1
for row in rows:
	if pid != row[0]:
		lines_event.append(line_e.strip()+'\n')
		lines_time.append(line_t.strip()+'\n')
		pid = row[0]
		time0 = row[2].timestamp()
		line_e = str(row[0]) + " "
		line_t = str(row[0]) + " "
	line_e += str(dl_dict[row[1]]) + " " 
	line_t += str( row[2].timestamp() - time0 ) + " "
lines_event.append(line_e.strip()+'\n')
lines_time.append(line_t.strip()+'\n')

with open('patients_paragraph_event.txt', 'w') as f:
	f.writelines(lines_event[1:])

with open('patients_paragraph_time.txt', 'w') as f:
	f.writelines(lines_time[1:])

# Diagnosis
# Dictionary for lab events
query = "SELECT DISTINCT p.SUBJECT_ID, dicd.ICD9_CODE from patients as p, diagnoses_icd as dicd WHERE p.SUBJECT_ID = dicd.SUBJECT_ID ORDER BY p.SUBJECT_ID"
cur.execute(query)
rows = cur.fetchall()

patient_dic = {}
pid = -1
arr = []
for row  in rows:
	if pid != row[0]:
		patient_dic[pid] = arr
		pid = row[0]
		arr = []
	arr.append(row[1])
patient_dic[pid] = arr

del patient_dic[-1]
with open('patient_diagnosis.json', 'w') as f:
	json.dump(patient_dic, f, indent=2)

conn.commit()
conn.close()
