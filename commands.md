<p>Einfügung eines Datensatzes in die Mainzelliste mithilfe des Benchmarking-Tools (-i: Spezifizieren des Dateipfades der einzufügenden Datei, -l=Anzahl der einzufügenden Zeilen, -u=URL der Mainzelliste-Instanz). <br>
Verwendungsbeispiel:</p>

```console
docker run -v "/$PWD:/root" docker.verbis.dkfz.de/pseudonymisierung/mainzelliste.client.cli:1.0.4 ADD_PATIENT -t=CSV --csv-separator=";" -i="./data/mut_test_records_10000.csv" -ids=pid -l=10000 -u=http://host.docker.internal:8080  -k=pleaseChangeMeToo -ii -out="output.csv" -s
```

<p>Export der Match-Ergebnisse aus der Datenbank der Mainzelliste in eine csv-Datei:</p>

```console
docker exec -it <ID_DOCKER_CONTAINER_DATABASE> psql -U mainzelliste mainzelliste -c "COPY(SELECT assignedpatient_patientjpaid,inputfieldsstring,bestMatchedWeight,bestmatchedpatient_patientjpaid,timestamp FROM public.idrequest ORDER BY timestamp ASC) TO STDOUT WITH CSV HEADER" > db_check.csv
```

<p>Export der in die Mainzelliste eingefügten Patientendaten in eine csv-Datei:</p>

```console
docker exec -it <ID_DOCKER_CONTAINER_DATABASE> psql -U mainzelliste mainzelliste -c "COPY(SELECT * FROM public.patient) TO STDOUT WITH CSV HEADER" > patient_records.csv
```