module tensor
contains
  subroutine sum(V1, V2, T)
    implicit none
    real(8), intent(in)  :: V1(:)
    real(8), intent(in)  :: V2(:)
    real(8), intent(out) :: T(size(V1))
    integer :: i
    integer :: n

    n = size(V1)
    if (size(V2) /= n) then
        print *, "Error: Vectors must have the same length"
        stop
    end if

    do i = 1, n
        T(i) = V1(i) + V2(i)
    end do
  end subroutine sum
end module tensor
