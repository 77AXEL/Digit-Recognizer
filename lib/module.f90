module tensor
contains
  subroutine sum(V1, V2, T, n)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in)  :: V1(n)
    real(8), intent(in)  :: V2(n)
    real(8), intent(out) :: T(n)
    integer :: i

    do concurrent (i = 1:n)
        T(i) = V1(i) + V2(i)
    end do
  end subroutine sum
end module tensor