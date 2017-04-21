#!/bin/bash

q1='{"channels": ["SINDI01-RIQM-DCP10:FOR-PHASE-AVG", "S10CB01-RBOC-DCP10:FOR-PHASE-AVG"], "fields": ["pulseId", "globalSeconds", "globalDate", "value"], "range": {"endDate": "2016-09-06T08:31:10.912526+02:00", "startDate": "2016-09-06T08:29:30.864924+02:00"}}'


#'{"channels":["S10CB01-RBOC-DCP10:FOR-PHASE-AVG", "S10CB01-RBOC-DCP10:FOR-PHASE-AVG"], "fields":["pulseId","globalSeconds","globalDate","value"], "range":{"startDate":"2016-08-16T07:37:35.519425", "endDate":"2016-08-16T07:39:15.519499"}}'

curl  -H "Content-Type: application/json" -X POST -d \
      '{"channels": ["SINDI01-RIQM-DCP10:FOR-PHASE-AVG", "S10CB01-RBOC-DCP10:FOR-PHASE-AVG"], "fields": ["pulseId", "globalSeconds", "globalDate", "value"], "range": {"endDate": "2016-09-07T16:00:53.424531+02:00", "startDate": "2016-09-07T15:59:13.423690+02:00"}}'\
      http://data-api.psi.ch/sf/query | python -m json.tool > out1.json 

curl  -H "Content-Type: application/json" -X POST -d\
      '{"channels": ["SINDI01-RIQM-DCP10:FOR-PHASE-AVG", "S10CB01-RBOC-DCP10:FOR-PHASE-AVG"], "fields": ["pulseId", "globalSeconds", "globalDate", "value"],"range": {"endSeconds":"1473256847.667999983", "startSeconds": "1473256753.424999952"}}'\
      http://data-api.psi.ch/sf/query | python -m json.tool > out2.json
