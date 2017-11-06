import numpy as np

def numtorn(cnt,subone=False):
	"""
	Convert cnt to a roman numeral
	if subone is True, the code assumes cnt=0 is the neutral state
	if subone is False, the code assumes cnt=1 is the neutral state
	"""
	sb = 0
	if subone: sb = 1
	if (cnt > 55) or (cnt+sb < 1):
		print "ERROR :: Roman numerals > 30 (and < 1) are not implemented"
	if cnt == 1-sb:
		return "I"
	elif cnt == 2-sb:
		return "II"
	elif cnt == 3-sb:
		return "III"
	elif cnt == 4-sb:
		return "IV"
	elif cnt == 5-sb:
		return "V"
	elif cnt == 6-sb:
		return "VI"
	elif cnt == 7-sb:
		return "VII"
	elif cnt == 8-sb:
		return "VIII"
	elif cnt == 9-sb:
		return "IX"
	elif cnt == 10-sb:
		return "X"
	elif cnt == 11-sb:
		return "XI"
	elif cnt == 12-sb:
		return "XII"
	elif cnt == 13-sb:
		return "XIII"
	elif cnt == 14-sb:
		return "XIV"
	elif cnt == 15-sb:
		return "XV"
	elif cnt == 16-sb:
		return "XVI"
	elif cnt == 17-sb:
		return "XVII"
	elif cnt == 18-sb:
		return "XVIII"
	elif cnt == 19-sb:
		return "XIX"
	elif cnt == 20-sb:
		return "XX"
	elif cnt == 21-sb:
		return "XXI"
	elif cnt == 22-sb:
		return "XXII"
	elif cnt == 23-sb:
		return "XXIII"
	elif cnt == 24-sb:
		return "XXIV"
	elif cnt == 25-sb:
		return "XXV"
	elif cnt == 26-sb:
		return "XXVI"
	elif cnt == 27-sb:
		return "XXVII"
	elif cnt == 28-sb:
		return "XXVIII"
	elif cnt == 29-sb:
		return "XXIX"
	elif cnt == 30-sb:
		return "XXX"
	elif cnt == 31-sb:
		return "XXXI"
	elif cnt == 32-sb:
		return "XXXII"
	elif cnt == 33-sb:
		return "XXXIII"
	elif cnt == 34-sb:
		return "XXXIV"
	elif cnt == 35-sb:
		return "XXXV"
	elif cnt == 36-sb:
		return "XXXVI"
	elif cnt == 37-sb:
		return "XXXVII"
	elif cnt == 38-sb:
		return "XXXVIII"
	elif cnt == 39-sb:
		return "XXXIX"
	elif cnt == 40-sb:
		return "XL"
	elif cnt == 41-sb:
		return "XLI"
	elif cnt == 42-sb:
		return "XLII"
	elif cnt == 43-sb:
		return "XLIII"
	elif cnt == 44-sb:
		return "XLIV"
	elif cnt == 45-sb:
		return "XLV"
	elif cnt == 46-sb:
		return "XLVI"
	elif cnt == 47-sb:
		return "XLVII"
	elif cnt == 48-sb:
		return "XLVIII"
	elif cnt == 49-sb:
		return "XLIX"
	elif cnt == 50-sb:
		return "L"
	elif cnt == 51-sb:
		return "LI"
	elif cnt == 52-sb:
		return "LII"
	elif cnt == 53-sb:
		return "LIII"
	elif cnt == 54-sb:
		return "LIV"
	elif cnt == 55-sb:
		return "LV"
	return None

def rntonum(cnt):
	"""
	Convert a roman numeral to the ion charge number
	"""
	if cnt == "I":
		return 0
	elif cnt == "II":
		return 1
	elif cnt == "III":
		return 2
	elif cnt == "IV":
		return 3
	elif cnt == "V":
		return 4
	elif cnt == "VI":
		return 5
	elif cnt == "VII":
		return 6
	elif cnt == "VIII":
		return 7
	elif cnt == "IX":
		return 8
	elif cnt == "X":
		return 9
	elif cnt == "XI":
		return 10
	elif cnt == "XII":
		return 11
	elif cnt == "XIII":
		return 12
	elif cnt == "XIV":
		return 13
	elif cnt == "XV":
		return 14
	return None

def numtoelem(cnt,subone=False):
	"""
	Convert cnt to an element
	"""
	sb = 0
	if subone: sb = 1
	if (cnt > 30) or (cnt < 1):
		if cnt not in [36,42,54]:
			print "ERROR :: Elements with atomic number > 30 (and < 1) are not implemented"
	if cnt == 1-sb:
		return "H"
	elif cnt == 2-sb:
		return "He"
	elif cnt == 3-sb:
		return "Li"
	elif cnt == 4-sb:
		return "Be"
	elif cnt == 5-sb:
		return "B"
	elif cnt == 6-sb:
		return "C"
	elif cnt == 7-sb:
		return "N"
	elif cnt == 8-sb:
		return "O"
	elif cnt == 9-sb:
		return "F"
	elif cnt == 10-sb:
		return "Ne"
	elif cnt == 11-sb:
		return "Na"
	elif cnt == 12-sb:
		return "Mg"
	elif cnt == 13-sb:
		return "Al"
	elif cnt == 14-sb:
		return "Si"
	elif cnt == 15-sb:
		return "P"
	elif cnt == 16-sb:
		return "S"
	elif cnt == 17-sb:
		return "Cl"
	elif cnt == 18-sb:
		return "Ar"
	elif cnt == 19-sb:
		return "K"
	elif cnt == 20-sb:
		return "Ca"
	elif cnt == 21-sb:
		return "Sc"
	elif cnt == 22-sb:
		return "Ti"
	elif cnt == 23-sb:
		return "V"
	elif cnt == 24-sb:
		return "Cr"
	elif cnt == 25-sb:
		return "Mn"
	elif cnt == 26-sb:
		return "Fe"
	elif cnt == 27-sb:
		return "Co"
	elif cnt == 28-sb:
		return "Ni"
	elif cnt == 29-sb:
		return "Cu"
	elif cnt == 30-sb:
		return "Zn"
	elif cnt == 36-sb:
		return "Kr"
	elif cnt == 42-sb:
		return "Mo"
	elif cnt == 54-sb:
		return "Xe"
	else:
		print "ERROR :: could not find element", cnt
	return None

def calc_yprofs(ions,rates,elID):
	"""
	Y_(X^n+) = n(X^n+) / SUM_i{ n(X^i+) }
	"""
	Yprofs = np.zeros_like(rates)
	# Calculate the Yprof for each ion under consideration
	elems = dict({})
	for i in range(len(ions)):
		espl = ions[i].split()
		telem = espl[0]
		if telem not in elems.keys(): elems[telem] = [rntonum(espl[1])]
		else: elems[telem] += [rntonum(espl[1])]
	# Now sort the ionization levels
	ekeys = elems.keys()
	for i in range(len(ekeys)):
		elems[ekeys[i]] = np.sort(elems[ekeys[i]])
	# Calculate the Yprof for each element/ion stage
	for i in range(len(ekeys)):
		# Calculate the profile for the neutral state
		invrate = 1.0
		for j in range(1,1+len(elems[ekeys[i]])):
			iname = ekeys[i] + " " + numtorn(elems[ekeys[i]][-j]+1)
			idx = elID[iname][0]
			invrate *= rates[:,idx]
			invrate += 1.0
		# Set the neutral state
		Yprofs[:,elID[ekeys[i] + " I"][0]] = 1.0/invrate.copy()
		# Now calculate the first, then second rate etc.
		for j in range(1,len(elems[ekeys[i]])):
			iname = ekeys[i] + " " + numtorn(j)
			invrate /= rates[:,elID[iname][0]]
			sname = ekeys[i] + " " + numtorn(j+1)
			Yprofs[:,elID[sname][0]] = 1.0/invrate.copy()
	return Yprofs.copy()
