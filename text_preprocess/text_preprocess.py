data = open("WA_MUDSLIDE","r")
output = open("WA_mudslide_100","w")
contents=data.read()
l=contents.split("\n")
print(len(l))
print(l[0])
for each in l:
    each = each.split(", ")
    del each[-4:]
    output.write(each[0])
    output.write("\n")

