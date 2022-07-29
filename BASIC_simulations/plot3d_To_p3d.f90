program main
implicit real*8 (a-h, o-z)

!    real*8, dimension(:,:,:), allocatable:: x, y, z
!    real*8, dimension(:,:,:,:), allocatable:: q

    real*8, dimension(:,:), allocatable:: xc, zc, iblank
    real*8, dimension(:,:,:), allocatable:: qc
!



debug_mod = 0


    open(9, file="plot3dg.bin", form="unformatted")
    read(9)ng
    if (debug_mod .ne. 0)write(*,*)ng
    read(9)jdim, kdim
    if (debug_mod .ne. 0)write(*,*)jdim, kdim
    allocate( xc(jdim, kdim), zc(jdim, kdim), iblank(jdim,kdim) )
    allocate( qc(jdim, kdim, 4) )
    read(9)((xc(j,k), j=1,jdim), k=1,kdim),   &
           ((zc(j,k), j=1,jdim), k=1,kdim)
           !((iblank(j,k), j=1,jdim), k=1,kdim)
    close(9) 







    open(10,file="plot3dq.bin", form="unformatted", status="old") 
    read(10)nplots
    read(10)jdim, kdim 
    if (debug_mod .ne. 0)write(*,*)jdim, kdim
    read(10)xmach,alpha,reue,time
    if (debug_mod .ne. 0)write(*,*)xmach,alpha,reue,time
    read(10)(((qc(j,k,m),j=1,jdim),k=1,kdim),m=1,4)
    close(10)







    open(7,file="plot3dg.p3d", status="unknown") 
    write(7,*)jdim, kdim
    do k = 1, kdim
    do j = 1, jdim
    write(7,*)xc(j,k) 
    enddo
    enddo
    do k = 1, kdim
    do j = 1, jdim
    write(7,*)zc(j,k) 
    enddo
    enddo
    close(7)

    
    open(7,file="plot3dg_stats.p3d", status="unknown") 
    write(7,*)""
    write(7,*)"rho, rhou, rhov, rhoE"
    write(7,*)4
    !write(7,*)jdim, kdim
    
    do m=1,4

    do k = 1, kdim
    do j = 1, jdim
    write(7,*)qc(j, k, m) 
    enddo
    enddo

    enddo !do



    !open(8,file="test_solution_3D.plt")
    !write(8,*)"variables= x, z, rho"            
    !write(8,*)"zone f= point j=", jdim, " k=", kdim
    !do k = 1, kdim
    !    do j = 1, jdim
    !           write(8,*)xc(j,k), zc(j,k), qc(j,k,1)
    !    enddo
    !enddo
    !close(8)
    
    

end program
