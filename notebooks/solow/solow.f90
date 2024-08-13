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
