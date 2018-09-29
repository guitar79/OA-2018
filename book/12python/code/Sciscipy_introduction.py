from scilab import Scilab
sci = Scilab()
x = sci.rand(20, 20)
y = x*x.transpose()
y_inv = sci.inv(y)