import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HomeComponent } from './modules/home/home.component';
import { HeaderComponent } from './modules/shared/header/header.component';
import { FooterComponent } from './modules/shared/footer/footer.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MaterialModule } from './modules/shared/material/material.module';
import { SidenavComponent } from './modules/shared/sidenav/sidenav.component';
import { NavService } from './nav.service';
import { HttpClient, HttpClientModule, HttpHandler } from '@angular/common/http';
import { LinearRegressionComponent } from './modules/Supervised Learning/linear-regression/linear-regression.component';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { NoSanitizePipe } from './modules/Supervised Learning/linear-regression/linear-regression.utils';

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    HeaderComponent,
    FooterComponent,
    SidenavComponent,
    LinearRegressionComponent,
    NoSanitizePipe,
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    MaterialModule,
    HttpClientModule,
    FormsModule,
    ReactiveFormsModule,
    MatInputModule,
    MatFormFieldModule
  ],
  providers: [NavService],
  bootstrap: [AppComponent],
  exports: [MatFormFieldModule, MatInputModule]
})
export class AppModule { }
