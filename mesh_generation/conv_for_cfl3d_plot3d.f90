program main


    real*8, dimension(:,:,:), allocatable:: x, y, z
    character*100:: p3d_fileName 

    read(*,*)p3d_fileName
    open(7,file=trim(p3d_fileName)//".p3d") 
    read(7,*) ni, nj
    !write(*,*) ni, nj
    jdim = ni
    kdim = nj
    idim = 2

    allocate( x(idim, jdim, kdim), y(idim, jdim, kdim), z(idim, jdim, kdim) )

    do k = 1, kdim
    do j = 1, jdim
    read(7,*)tempx
    x(:,j,k) = tempx
    enddo
    enddo
    do k = 1, kdim
    do j = 1, jdim
    read(7,*)tempz
    z(:,j,k) = tempz
    enddo
    enddo
    close(7)
    
    y(1,:,:) = 0
    y(2,:,:) = -1

    if (1==1) then
    write(*,*)"yes"
    open(8,file=trim(p3d_fileName)//".plt")
    write(8,*)"variables= x, y, z"            
    write(8,*)"zone f= point,i=", 2, " j=", ni, " k=", nj
    do k = 1, kdim
        do j = 1, jdim
           do i = 1, idim
               write(8,*)x(i,j,k), y(i,j,k), z(i,j,k)
           !i=1
               !write(8,*)x(i,j,k), 0, z(i,j,k)
               !write(8,*)x(i,j,k), -1, z(i,j,k)
           enddo
        enddo
    enddo
    endif
    close(8)
    ngrid = 1
    open(9, file=trim(p3d_fileName)//".bin", form="unformatted")
    write(9)ngrid
    write(9)idim, jdim, kdim
    write(9)(((x(i,j,k), i=1,idim), j=1,jdim), k=1,kdim),   &
            (((y(i,j,k), i=1,idim), j=1,jdim), k=1,kdim),   &
            (((z(i,j,k), i=1,idim), j=1,jdim), k=1,kdim)

    close(9) 

end program
