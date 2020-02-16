function [d dy dh ddy ddh] = jf_checkgrad(fFn, x, e, hesDiag, verb)
% [d dy dh ddy ddh] = checkgrad(fFn, x, e, hesDiag, verb)
% checkgrad checks the derivatives in a function, by comparing them to finite
% differences approximations. The partial derivatives and the approximation
% are printed and the norm of the diffrence divided by the norm of the sum is
% returned as an indication of accuracy.
%
% N.B. for multi-dimensional outputs we assume:
%  dfdx = [ size(f) x size(x) ] 
%
% 2006-08-01 : modified to work with vector output functions also
% 2006-08-01 : modified to use function objects with a single argument, JF
%              modified to return diag hessian estiamte also
% $Id: checkgrad.m,v 1.16 2007-06-24 21:06:36 jdrf Exp $
if ( nargin < 3 || isempty(e) ) e=eps; end;
if ( nargin < 4 || isempty(hesDiag) ) hesDiag=false; end;
if ( nargin < 5 || isempty(verb) ) verb=0; end;
if ( hesDiag ) % get the partial derivatives dy
   if ( iscell(fFn) ) 
      [y dy ddy]=feval(fFn{1},x,fFn{2:end}); 
   else 
      [y dy ddy]=feval(fFn,x);  
   end;
else
   if ( iscell(fFn) ) 
      [y dy]=feval(fFn{1},x,fFn{2:end}); 
   else 
      [y dy]=feval(fFn,x);  
   end;
   ddy=zeros(size(x));
end
% shape to 2-d matrix of numel(x) x numel(f) == N.B. must have same #el as dy
dh =zeros(size(dy));
dh =reshape(dh,[prod(size(y)),prod(size(x))]); % grad is size(y) x size(x)
ddh=zeros(size(dh));
tx = x;
for j = 1:numel(x)
   fprintf('.');  
   tx(j) = x(j)+e;                               % perturb a single dimension
   if ( iscell(fFn) ) 
      y2=feval(fFn{1},tx,fFn{2:end}); 
   else 
      y2=feval(fFn,tx);  
   end;
   tx(j) = x(j)-e ;
   if ( iscell(fFn) ) 
      y1=feval(fFn{1},tx,fFn{2:end}); 
   else 
      y1=feval(fFn,tx);  
   end;
   tx(j) = x(j);                                 % reset it
   dh(:,j) = (y2(:) - y1(:))/(2*e);
   ddh(:,j)= (y2(:) - 2*y(:) + y1(:))/(e*e);     % diag hessian est
end
% reshape output to be the same as the one generated by the function
dh=reshape(dh,size(dy));
if ( hesDiag ) ddh=reshape(ddh,size(ddy)); end;
fprintf('\n');
if (verb > 0 ) 
   fprintf('[dy(:) dh(:)] =\n');
   disp([dy(:) dh(:)])                                      % print the vectors
   if ( hesDiag ) 
      fprintf('[ddy(:) ddh(:)] =\n'); disp([ddy(:) ddh(:)]); 
   end;
end;
% return norm of diff divided by norm of sum
d=[norm(dh(:)-dy(:))/norm(dy(:)) norm(dh(:)-dy(:))/norm(dh(:)+dy(:))];
if ( hesDiag ) 
   d=[d norm(ddh(:)-ddy(:))/norm(ddy(:)) norm(ddh(:)-ddy(:))/norm(ddh(:)+ddy(:))]; 
end;
if( verb>0 | nargout==0 ) 
   fprintf('dy: \t%0.6g \t%0.6g\nddy:\t%0.6g \t%0.6g\n',d)
end;
