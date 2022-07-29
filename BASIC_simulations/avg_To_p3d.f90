program iain
implicit real*8 (a-h, o-z)

!    real*8, dimension(:,:,:), allocatable:: x, y, z
!    real*8, dimension(:,:,:,:), allocatable:: q

    real*8, dimension(:,:,:), allocatable:: xc, yc, zc
    integer, dimension(:,:,:), allocatable:: iblank
    real*8, dimension(:,:,:,:), allocatable:: qc
!



debug_mod = 1


    open(9, file="cfl3d_avgg.p3d", form="unformatted")
    read(9)ng
    if (debug_mod .ne. 0)write(*,*)ng
    read(9)idim, jdim, kdim
    if (debug_mod .ne. 0)write(*,*)idim, jdim, kdim
    allocate( xc(jdim, kdim, idim),yc(jdim, kdim, idim), zc(jdim, kdim, idim), iblank(jdim,kdim,idim) )
    allocate( qc(jdim, kdim, idim, 5) )
    !read(9) text
    !write(*,*)text
              
           
         read(9)   (((xc(j,k,i),i=1,idim),j=1,jdim),k=1,kdim),&
                   (((yc(j,k,i),i=1,idim),j=1,jdim),k=1,kdim),&
                   (((zc(j,k,i),i=1,idim),j=1,jdim),k=1,kdim),&
                   (((iblank(j,k,i), i=1,idim),j=1,jdim),k=1,kdim)
    close(9) 



    



    open(10,file="cfl3d_avgq.p3d", form="unformatted", status="old") 
    read(10)nplots
    read(10)idim, jdim, kdim 
    if (debug_mod .ne. 0)write(*,*)idim, jdim, kdim
    read(10)xmach,alpha,reue,time
    if (debug_mod .ne. 0)write(*,*)xmach,alpha,reue,time
    read(10)((((qc(j,k,i,m),i=1,idim),j=1,jdim),k=1,kdim),m=1,5)
    close(10)







    open(7,file="plot3dg.p3d", status="unknown") 
    write(7,*)jdim, kdim
    do k = 1, kdim
    do j = 1, jdim
    write(7,*)xc(j,k,1) 
    enddo
    enddo
    do k = 1, kdim
    do j = 1, jdim
    write(7,*)zc(j,k,1) 
    enddo
    enddo
    close(7)

    
    open(7,file="plot3dg_stats.p3d", status="unknown") 
    write(7,*)""
    write(7,*)"rho, rhou, rhov, rhoE"
    write(7,*)4
    !write(7,*)jdim, kdim
    
    ! m=1 --- rho
    ! m=2 --- rho-U
    ! m=3 --- rho-V --- y direction -- 0
    ! m=4 --- rho-W
    ! m=5 --- rho-E 
    do m=1,2

    do k = 1, kdim
    do j = 1, jdim
    write(7,*)qc(j, k, 1, m) 
    enddo
    enddo

    enddo !do

    do m=4,5

    do k = 1, kdim
    do j = 1, jdim
    write(7,*)qc(j, k, 1, m) 
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
