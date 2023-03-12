import matplotlib
print(matplotlib.__version__)
import normalize

# Create an instance of the Normalize class
p = normalize.Normalize('fruits.jpg')
p = p.phonenumber_filter()
print(p)
