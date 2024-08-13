---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Solow--Swan: Pure Python vs Fortran vs Numba

----
#### John Stachurski (August 2024)

----

Here's a pure Python version

```{code-cell} ipython3
def solow(k0, α=0.4, δ=0.1, s=0.3, n=10_000_000):
    k = k0
    for i in range(n-1):
        k = s * k**α + (1 - δ) * k
    return k
```

Let's see how long it takes to run.

```{code-cell} ipython3
%time solow(0.2)
```

Here's a Fortran version

```{code-cell} ipython3
%%file solow.f90
module solow_module
  implicit none
  integer, parameter :: dp = kind(0.d0)
contains
  pure function solow(k0, n) result(k)
    integer, intent(in) :: n
    real(dp), intent(in) :: k0
    real(dp) :: k
    integer :: i

    k = k0
    do i = 1, n - 1
      k = 0.3_dp * k**0.4_dp + (1 - 0.1_dp) * k
    end do
  end function solow
end module solow_module

program main
  use solow_module
  implicit none
  integer :: n = 10000000
  real(dp) :: start, finish, k 
  real(dp) :: k0 = 0.2_dp

  call cpu_time(start)
  k = solow(k0, n)
  call cpu_time(finish)

  print *, 'Elapsed time in seconds = ', finish - start
  print *, 'k = ', k
end program main
```

Let's compile it:

```{code-cell} ipython3
!gfortran -o out solow.f90
```

And now let's run it:

```{code-cell} ipython3
!./out
```

Next we'll try a Numba version

```{code-cell} ipython3
from numba import jit

@jit
def solow(k0, α=0.4, δ=0.1, s=0.3, n=10_000_000):
    k = k0
    for i in range(n-1):
        k = s * k**α + (1 - δ) * k
    return k
```

Let's see how long it takes to run.

```{code-cell} ipython3
%time solow(0.2)
```

```{code-cell} ipython3

```
