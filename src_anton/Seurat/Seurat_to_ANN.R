# Define the grid for x and y values
x <- seq(-3, 3, length.out = 30)
y <- seq(-3, 3, length.out = 30)

# Create a meshgrid (grid of all combinations of x and y)
grid <- expand.grid(x = x, y = y)

# Define a bivariate normal distribution (independent variables)
z <- dnorm(grid$x) * dnorm(grid$y)  # Normal distribution on both axes

# Reshape z into a matrix suitable for the 'persp' function
z_matrix <- matrix(z, nrow = length(x), ncol = length(y))

# Create a 3D plot using persp
persp(x, y, z_matrix, 
      main = "3D Normal Distribution", 
      xlab = "X", 
      ylab = "Y", 
      zlab = "Density", 
      col = "lightblue", 
      theta = 30, phi = 30)
