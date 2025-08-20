## How to solve a Linear System of Equations:
1. Write the system of equations in matrix form.
2. Use Gaussian elimination to transform the matrix into row echelon form.
3. Use back substitution to find the solution.

### 1. Matrix form:
suppose we are given below equations:

```
2x + 3y + z = 10
4x - y + 2z = 11
x + y = 6
```

to represent the system of linear equations in matrix form:

```
| 2  3  1 | | x |   | 10 |
| 4 -1  2 | | y | = | 11 |
| 1  1  0 | | z |   |  6 |
```

You can notice three things here:
1. we have taken coefficients of each variable and lined them up in a matrix. we will call it matrix A.
2. we have placed all the variables in a vector. we will call it vector X.
3. we have placed all the constants on the right side in a vector. we will call it vector b.

because we have given proper markations, we can write the system of linear equations as:

```
AX = b
```

where A is the coefficient matrix, X is the vector of variables, and b is the matrix of constants.

This way of representing a system of linear equations is called the matrix form. It is a compact and efficient way to represent a system of linear equations. It is also useful for solving systems of linear equations using matrix operations.

### 2. Row Elimination:
Row elimination is a method used to transform a matrix into row echelon form. It involves performing elementary row operations on the matrix to eliminate the entries below the leading coefficient in each row.

Let's define Row Echelon Form (REF) in a matrix:
1. All non zero rows are above any row with all zeros.
2. The leading entry (first non-zero number from the left - also called pivot) is always on the right side of the leading entry of the row above it.
3. All the entries in a column below the pivot should be zeros.

***Side note:*** There is a strictier version of REF called Reduced Row Echelon Form (RREF). It has two more conditions:
1. The pivot must be 1 in each row.
2. The pivot should be only non-zero entry in  it's column.

Now that we have defined REF, let's discuss how to transform a matrix into REF using row operations.

If we have a system of equation like below:

```
2x + 3y + 2z = 10
4x + 5y + 2z = 11
2x + y = 6
```

we try to eliminate one of the variables to reduce the system to a simpler form. Let's eliminate x from the second and third equations. To do so, we will perform the following row operations:
- Multiply the first row by 2 and subtract it from the second row.
- Multiply the first row by 1 and subtract it from the third row.

```
2x + 3y + 2z = 10
0x - y - 2z = -9
0x - 2y - 2z = -4
```

We have reduced the system to a simpler form. we can still remove y from the third equation. to do that:
- Multiply the second row by 2 and subtract it from the third row.

```
2x + 3y + 2z = 10
0x - y - 2z = -9
0x + 0y + 2z = 14
```

Now that we have eliminated y from the third equation, we can see that z = 7.
we can now start backfilling this value in above equations, to get the values of y and x.

this gives us:

```
x = 11/2
y = -5
z = 7
```

this is the conventional method that we used to learn in high school. Now, let's take a look at the matrix form of this system.
if we want to represent the system of linear equations in matrix form, we can write it as:

```
| 2  3  2 | | x |     | 10 |
| 4  5  2 | | y |  =  | 11 |
| 2  1  0 | | z |     |  6 |
     |        |         |
     A        X         b
```

because we are only interested in the matrix form of the system, more specifically, in the coefficients of the variables, we can ignore the values of x, y, and z.

Now, based on the previous examples, we can get the matrix after first step of eliminations, which looks like this:

```
| 2  3  2 |
| 0 -1 -2 |
| 0 -2 -2 |
```

Let's say we want to get this matrix by doing matrix multiplication of A with another matrix E. Here, Matrix E is called Elementary matrix (because it is used for the purpose of elimination).

```
  | 2  3  2 |   | 2  3  2 |
E | 4  5  2 | = | 0 -1 -2 |
  | 2  1  0 |   | 0 -2 -2 |
```
We want to find E such that (row 2, column 1) becomes 0. (as per the conditions of REF). To do so, we have to make sure that row 1 remains same, where as row 2 and row 3 changes based on step 1 of elimination. which means:
- Multiply the first row by 2 and subtract it from the second row.
- Multiply the first row by 1 and subtract it from the third row.

If assume E starts as an Identity matrix, then we can apply the above steps to get the matrix E.

before applying the steps, we have:

```
| 1  0  0 | | 2  3  2 |
| 0  1  0 | | 4  5  2 |
| 0  0  1 | | 2  1  0 |
```

after applying the steps, we have:

```
|  1  0  0 | | 2  3  2 |   | 2  3  2 |
| -2  1  0 | | 4  5  2 | = | 0 -1 -2 |
| -1  0  1 | | 2  1  0 |   | 0 -2 -2 |
```

so, we found an Elementary matrix E such that we get our elimination matrix.

During this matrix multiplication, we were making sure that we find a pivot for row 2. to do that, we had to make sure that (row 2, column 1) is zero, since row 1 has a non-zero value at column 1. Because our pivot for this elimination is (row 2, column 1), we can mark out Elementary matrix E as E<sub>21</sub>.

```
                 |  1  0  0 |
E<sub>21</sub> = | -2  1  0 |
                 | -1  0  1 |
```

same way, we now want to find E<sub>32</sub> to make sure that (row 3, column 2) is zero. To do so, we have to make sure that row 1 and 2 remains same, where as row 3 changes based on step 2 of elimination. which means:
- Multiply the second row by 2 and subtract it from the third row.

before applying the step, we have:

```
| 1  0  0 | | 2  3  2 |   | 2  3  2 |
| 0  1  0 | | 0 -1 -2 | = | 0 -1 -2 |
| 0  0  1 | | 0 -2 -2 |   | 0  0  2 |
```

after applying the step, we have:

```
| 1  0  0 | | 2  3  2 |   | 2  3  2 |
| 0  1  0 | | 0 -1 -2 | = | 0 -1 -2 |
| 0 -2  1 | | 0 -2 -2 |   | 0  0  2 |
```

so, E<sub>32</sub> will be:

```
                 |  1  0  0 |
E<sub>32</sub> = |  0  1  0 |
                 |  0 -2  1 |
```

now we can write:

```
(E<sub>32</sub>(E<sub>21</sub>A)) = REF(A)
```
