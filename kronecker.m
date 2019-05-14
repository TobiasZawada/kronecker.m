%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Date: 05/09/19
%% Octave-Version: 2019-05-14
%%
%% Authors: Tobias Zawada (same person as former Tobias Naehring) (http://www.tn-home.de),
%%          maybe you;-))
%%
%% License: GNU Lesser General Public License
%% (For details see http://www.gnu.org/copyleft/lesser.html)
%%
%% Bug reports to: (i at tn-home.de)
%%
%% Description:
%%
%% Run this script with
%%
%% octave> run full/path/to/kronecker.m
%%
%% Let A, B be two real M-by-N matrices (with natural numbers M,N). The
%% main routine
%%
%% [At, Bt, U, V, m, n, nf, ni] = pencil_struct(A,B,eps)
%%
%% computes a backward stable transformation Bt*s - At = U*( B*s - A )*V
%% where (Bt*s - At) is in lower block triangular form
%%
%% [	Bn*s - An,		zeros,		zeros,		zeros;
%%		X,	    Bf*s - Af,		zeros,		zeros;
%%	        X,	            X,	    Bi*s - Ai,		zeros;
%%		X,		    X		    X,	     Be*s - Ae]
%%
%% The entries X stand for arbitrary block matrices, and diagonal blocks
%% are structured as follows:
%%
%% Bn*s - An is a singular pencil with the same Kronecker column indices as
%% B*s - A. Its dimension is (n-nf)-by-(m-nf).
%%
%% Bf*s - Af is a regular nf-by-nf pencil with the same finite
%% eigenvalues as B*s-A.
%%
%% Bi*s - Ai is a regular ni-by-ni pencil with the same infinite eigenvalues as
%% B*s-A
%%
%% Be*s - Ae is a singular pencil with the same Kronecker row indices as
%% B*s-A. Its dimension is (N-n-ni)-by-(M-m-mi).
%%
%% As it is clear from the above description the Sub-pencil
%%
%% [	Bn*s - An,		zeros;
%%		X,	    Bf*s - Af]
%%
%% has the dimensions [m,n].
%%
%%
%% Some auxiliary routines are also provided with this file.
%%
%% The algorithm is taken from
%% [P. Van Dooren: The Computation of Kronecker's Canonical Form of a
%% Singular Pencil. LINEAR ALGEBRA AND ITS APPLICATIONS 27:103-140 (1979)].
%%
%% TODO:
%%
%% 1) Better documentation. Especially, better documentation of the routines
%% pencil_colstruct and pencil_rowstruct. (These two routines do
%% more than what is necessary for pencil_struct. They reveal the
%% complete Kronecker structure information.)
%%
%% 2) More comprehensive testing.
%%
%% 3) There are some performance relevant issues scattered throughout the
%% program text. Maybe these should be fixed some day. (If there is a real
%% demand for it.)
%%
%% Changes:
%% 05/09/20:
%%  Added pencil_contr(A,B,E):
%%  Computation of the controllable subspace of the pair (E*s-A,B) of a
%%  descriptor system.
%% 
%% 05/09/20:
%%  pencil_struct: Treat the case that E*s-A has no infinite eigenvalues
%%  and no column Kronecker indexes.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

0; %% This is a script file!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% [Q,r,C] = left_colcomp(A,eps)
%% Determines the rank r of a general matrix A
%% and returns the factors C, Q of a decomposition of type
%% A = [ C, zeros(m,n-r) ]   * Q''
function [Q,r,C] = left_colcomp(A,eps = sqrt(eps))
  [ U, S, Q ] = svd( A );
  r = nnz(diag(S) > eps);
  C = A * Q(:,1:r);
endfunction;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% [Q,C,r] = bottom_rowcomp(A,eps)
%% Determines the rank r of a general matrix A
%% and returns the factors Q, C of a decomposition of type
%% A = Q * [ zeros(m-r, n); C ]
function [Q,r,C] = bottom_rowcomp(A,eps=sqrt(eps))
  [ U, S, V ] = svd( A );
  r = nnz(diag(S) > eps);
  Q = [ U(:,r+1:size(A,1))'; U(:,1:r)' ];
  if argn(1) == 3
    C =  U(1:r,:) * A;
  end;
endfunction;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% [A, B, U, V, l, m, n, r, s] = pencil_colstruct( A, B, eps, U, V )
%% Computes the Kronecker column structure of a matrix pencil sB-A.
%% The arguments eps, U, and V are optional.
%% eps is the precision for the decomposition. Matrix elements smaller
%% than eps are regarded as numerical zeros.
%%
%% It may be that A and B already stem from a transformation
%% 
%% A = U * A_original * V
%% B = U * B_original * V
%%
%% If you specify U and V as arguments then these transformations are
%% continued to the transformation of s B_original-A_original to the
%% Kronecker column structure.
function [A, B, U, V, l, m, n, r, s] = pencil_colstruct( A, B, eps=sqrt(eps), U=eye(size(A,1)), V=eye(size(A,2)))
  if any(size(A) != size(B))
    error("Error in pencil_colstructure(A,B,eps):"+...
	  "A should have the same size as B.");
  end;
  [M,N] = size(A);

  m = M; n = N;
  %% Initialization to avoid errors for the case m == 0 || n == 0
  s = [];
  r = [];
  for l=0:size(A,1)

    if n == 0 || m == 0
      return;
    end;

    %% The `p' in `Up' stands for `partial transformation' and
    %% rB stands for the rank of B.
    [ Vp, rB ] = left_colcomp(B(1:m,1:n),eps);
    s(l+1) = n-rB;

    if rB == n
      return;
    end;

    A(:,1:n) = A(:,1:n)*Vp; %% Transform A the same way as B -> Bc
    B(:,1:n) = B(:,1:n)*Vp; %% Transform also the remainder of B.
    %% Clean up B: (TODO: This could be done in a more effective way.)
    B(1:m,rB+1:n) = zeros(m,s(l+1));

    %% Row compress the part of A in the kernel of B:
    [ Up, rA ] = bottom_rowcomp(A(1:m,rB+1:n),eps);
    r(l+1) = rA;
    A(1:m,:) = Up * A(1:m,:);
    B(1:m,1:n) = Up * B(1:m,1:n);
    %% Clean up A: (TODO: This could be done in a more effective way.)
    A(1:m-rA,rB+1:n) = zeros(m-rA,s(l+1));
    
    U(1:m,:) = Up * U(1:m,:);
    V(:,1:n) = V(:,1:n) * Vp;

    m = m - rA;
    n = rB;
  end;
  error("Too many iterations in pencil_structure1.\n"+...
	"This should never happen!");
endfunction;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The per-transposed algorithm is implemented in the lazy way:
%% Just work on the per-tranposed matrices.
%% 
%% That has two main advantages:
%% 1) One only needs to maintain the actual algorithm
%%    `pencil_colstruct'. (This implementation is less error-prone.)
%% 2) We avoid the ugly indexes for the per-transposed algorithm.
%% And one disadvantage:
%%    This implementation suffers from a slight loss in performance.
%%    Maybe, that becomes serious for larger problems.
function [A, B, U, V, l, m, n, r, s] = pencil_rowstruct( A, B, eps = sqrt(eps), U = eye(size(A,1)), V = eye(size(A,2)))
  APT = A(end:-1:1,end:-1:1)';
  BPT = B(end:-1:1,end:-1:1)';

  UPT = V(end:-1:1,end:-1:1)';
  VPT = U(end:-1:1,end:-1:1)';
  [APT, BPT, UPT, VPT, l, n, m, r, s] = pencil_colstruct(APT, BPT, eps, UPT, VPT);

  A = APT(end:-1:1,end:-1:1)';
  B = BPT(end:-1:1,end:-1:1)';
  V = UPT(end:-1:1,end:-1:1)';
  U = VPT(end:-1:1,end:-1:1)';
endfunction;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% [At, Bt, U, V, m, n, nf, ni] = pencil_struct(A,B,eps)
%%
%% computes a backward stable transformation Bt*s - At = U*( B*s - A )*V
%% where (Bt*s - At) is in lower block triangular form
%%
%% [	Bn*s - An,		zeros,		zeros,		zeros;
%%		X,	    Bf*s - Af,		zeros,		zeros;
%%	        X,	            X,	    Bi*s - Ai,		zeros;
%%		X,		    X		    X,	     Be*s - Ae]
%%
%% The entries X stand for arbitrary block matrices, and diagonal blocks
%% are structured as follows:
%%
%% Bn*s - An is a singular pencil with the same Kronecker column indices as
%% B*s - A. Its dimension is (n-nf)-by-(m-nf).
%%
%% Bf*s - Af is a regular nf-by-nf pencil with the same finite
%% eigenvalues as B*s-A.
%%
%% Bi*s - Ai is a regular ni-by-ni pencil with the same infinite eigenvalues as
%% B*s-A
%%
%% Be*s - Ae is a singular pencil with the same Kronecker row indices as
%% B*s-A. Its dimension is (N-n-ni)-by-(M-m-mi).
%%
%% As it is clear from the above description the Sub-pencil
%%
%% [	Bn*s - An,		zeros;
%%		X,	    Bf*s - Af]
%%
%% has the dimensions [m,n].
%%
%% blocks of B with rows <= n and columns <= m have full column rank.
%% n_f : number of rows/columns of the matrix pencil with finite
%% eigenvalues
%% n_i : number of rows/columns of the matrix pencil with infinite
%% eigenvalues
function [A, B, U, V, m, n, n_f, n_i] = pencil_struct(A,B,eps = sqrt(eps))
  [A, B, U, V, l, m, n] = pencil_colstruct(A, B,eps);
  %% m_col and n_col are the dimensions of the upper left block of B
  %% with full column rank.
  %% Next strategy: Divide and conquere.
  [ Afc, Bfc, Ufc, Vfc, l, n_f ] = pencil_rowstruct(...
      A(1:m,1:n),...
      B(1:m,1:n),...
      eps);

  
  U(1:m,:) = Ufc*U(1:m,:);

  if m < size(A,1)
    A(1:m,1:n) = Afc;
    B(1:m,1:n) = Bfc;
    A(m+1:end,1:n) = A(m+1:end,1:n)*Vfc;
    B(m+1:end,1:n) = B(m+1:end,1:n)*Vfc;
    V(:,1:n) = V(:,1:n) * Vfc;
    
    
    [ Asc, Bsc, Usc, Vsc, l, m_K ] = pencil_rowstruct(...
	A(m+1:end,n+1:end),...
	B(m+1:end,n+1:end),...
	eps);
    
    A(m+1:end,n+1:end) = Asc;
    B(m+1:end,n+1:end) = Bsc;
    A(m+1:end,1:n) = Usc*A(m+1:end,1:n);
    B(m+1:end,1:n) = Usc*B(m+1:end,1:n);
    
    U(m+1:end,:) = Usc * U(m+1:end,:);
    V(:,n+1:end) = V(:,n+1:end)*Vsc;
  else %% m = size(A,1) (no infinite eigenvalues and no column Kronecker
       %% indexes)
    m_K = 0;
  end;
  
  n_i = size(A,1)-m-m_K;

endfunction;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% An application of the pencil algorithm:
%% Computation of the controllable subspace of a pair ( s*E-A, B ).
%% Where s*E-A is a REGULAR n-by-n pencil and B is a n-by-p
%% matrix. More exactly:
%% [ U, V, Et, At, Bt, c ] = pencil_contr( A, B, E, eps )
%% computes orthogonal transformations U and V such that (s*Et-At, Bt)
%% with Et = U*E*V, At = U*A*V, Bt = U*B
%% has the following structure:
%%
%% [Et*s-At, Bt] = 
%% [ En*s-An,   zeros, zeros;
%%   XXXXXXX, Er*s-Ar, zeros;
%%   XXXXXXX, Ec*s-Ac;    Bc].
%%
%% The controllable part is the subsystem
%%
%%   [ Er*s-Ar, zeros;
%%    Ec*s-Ac;    Bc].
%%
%% The square matrix [ Er*s-Ar; Ec*s-Ac ]; has c columns/rows.
%% Therefore, the last c rows of V span the controllable subspace of
%% (E*s-A,B).
%%
%% See [Van Dooren: The Generalized Eigenstructure Problem in Linear
%% System Theory. IEEE TRANSACTIONS ON AUTOMATIC CONTROL, VOL. AC-26,
%% NO. 1, FEBRUARY 1981].
function [ U, V, E, A, B, c ] = pencil_contr( A, B, E, eps = sqrt(eps) )
  
  n = size(A,1);
  
  %% Compress B:
  [U, rB, Bc] = bottom_rowcomp( B, eps );

  if rB == size(A,1)
    return;
  end;
  
  B = [ zeros(n-rB,size(Bc,2)); Bc ];
  
  A = U*A;
  E = U*E;
  
  [ AK, EK, UK, V, mK, nK, nf, ni ] = ...
      pencil_struct( A(1:end-rB,:), E(1:end-rB,:), eps);
  
  U(1:n-rB,:) = UK * U(1:n-rB,:);
  
  c = n - nK - ni;
  
  A(1:end-rB,:) = AK;
  E(1:end-rB,:) = EK;
  
  A(end-rB+1,:) = A(end-rB+1,:)*V;
  E(end-rB+1,:) = E(end-rB+1,:)*V;
endfunction
