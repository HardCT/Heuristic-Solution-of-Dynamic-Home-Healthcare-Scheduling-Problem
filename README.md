# Heuristic Solution of Dynamic Home Healthcare Scheduling Problem

This code implements Capacity and Distance Heuristic(CH & DH) solution of dynamic scheduling problem of home healthcare, which is a reproduce of article https://doi.org/10.1080/19488300.2010.549818, with minor adjustments on rules of scheduling.

## Main changes

1. Omit setting the second level of DH as the backup level of CH.
2. Omit the rule "first visit in 24 hours after requesting".
3. Assume patients make appointment before the start of one week, thus no situation like "patient is available on Monday, but now is Tuesday, so the appoitment start at next week".

## Parameters setting

1. Single experiment at line 425 or repeating experiment since line 428.
2. Size of region, number of patients per week and distribution of patients' location(U, R, UC, R) at line 425 or 431.
3. Start and end time with the length of single time slot at line 19,20 and 21.
4. Nurse's home location at line 393.

## Result

No significant advantage of patient acceptance rate on CH over DH, which does not fully reveal experiment outcome of original article, may due to the changes of rules.
