program main


    !real*8, dimension(:,:,:), allocatable:: x, y, z, vol
    real*8, dimension(:,:,:,:), allocatable:: si, sj, sk

    character*100:: fileName
    jdim=129
    kdim=129
    allocate( si(jdim,kdim,1,4), sj(jdim,kdim,1,4), sk(jdim,kdim,1,4))
    write(*,*)"Input airfoil name for metrics, w/o (_metric.bin): "
    read(*,*)fileName
    !write(*,*)"read grid done"
    open(10, file="./metric_vol_cfl3d/"//trim(fileName)//"_metric.bin", form="unformatted", status="old", action="read")


    do k = 1,kdim
    do j = 1,jdim
    i = 1
    read(10) &
    si(j,k,i,4), &
    sj(j,k,i,1),sj(j,k,i,3),sj(j,k,i,4), &
    sk(j,k,i,1),sk(j,k,i,3),sk(j,k,i,4)
    !read(10)vol(j,k,i), &
    !si(j,k,i,1),si(j,k,i,2),si(j,k,i,3),si(j,k,i,4), &
    !sj(j,k,i,1),sj(j,k,i,2),sj(j,k,i,3),sj(j,k,i,4), &
    !sk(j,k,i,1),sk(j,k,i,2),sk(j,k,i,3),sk(j,k,i,4)
    enddo
    enddo
    close(10)

    open(7,file="./train_metric_p3d/"//trim(fileName)//"_metric.p3d", status="unknown") 
    write(7,*)""
    write(7,*)"si4, sj1, sj3, sj4, sk1, sk3, sk4"
    write(7,*)7
    !write(7,*)jdim, kdim
    
    i=1
    do k = 1, kdim-1
    do j = 1, jdim-1
    write(7,*)si(j, k, i, 4) 
    enddo
    enddo
    do k = 1, kdim-1
    do j = 1, jdim-1
    write(7,*)sj(j, k, i, 1) 
    enddo
    enddo
    do k = 1, kdim-1
    do j = 1, jdim-1
    write(7,*)sj(j, k, i, 3) 
    enddo
    enddo
    do k = 1, kdim-1
    do j = 1, jdim-1
    write(7,*)sj(j, k, i, 4) 
    enddo
    enddo
    do k = 1, kdim-1
    do j = 1, jdim-1
    write(7,*)sk(j, k, i, 1) 
    enddo
    enddo
    do k = 1, kdim-1
    do j = 1, jdim-1
    write(7,*)sk(j, k, i, 3) 
    enddo
    enddo
    do k = 1, kdim-1
    do j = 1, jdim-1
    write(7,*)sk(j, k, i, 4) 
    enddo
    enddo
    close(7)



end program
