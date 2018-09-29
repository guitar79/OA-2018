%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  LaTeX document class of the thesis of the Gyeonggi Science High School   %%%
%%%  Last edition 2015.11.13 by Chinook Mok                                   %%%
%%%  Continously being modified by gshslatexintro after 2016.02.02.           %%%
%%%  Check the latest version at : latex.gs.hs.kr                             %%%
%%%  Also refer to https://www.facebook.com/gshstexsociety                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ProvidesClass{book_main}

\ifxetex
\LoadClass[11pt]{article}
\else
\usepackage[nonfrench]{kotex}
\LoadClass[pdftex, 11pt]{article}
\fi 

\usepackage[margin=10pt,font=small,labelfont=bf]{caption}
\usepackage{subcaption}
%\usepackage{float}
%\floatstyle{plaintop}
%\restylefloat{table}
\usepackage[tableposition=top]{caption}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% caption format %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%\renewcommand{\figurename}{Fig.}
\captionsetup[figure]{%
	font=small,labelsep=period,singlelinecheck=off,aboveskip=0pt,% 
	belowskip=5pt, margin=0pt,subrefformat=parens%
}
\captionsetup[table]{%
	font=small,labelsep=period,singlelinecheck=off,aboveskip=5pt,% 
	margin=0pt%
}
\captionsetup[equation]{%
	font=small,labelsep=period,singlelinecheck=off,aboveskip=5pt,% 
	margin=0pt%
}
